import torch.nn as nn
from mixed_op import MixedOp_first, MixedOp_second
from mode import Mode
import torch

import torch.nn.functional as F

from typing import Union, Callable, Optional


class Conv2D(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 kernel_size: Union[tuple, list],
                 stride: Union[tuple, list] = (1, 1),
                 use_bias: bool = True,
                 activation: Optional[Callable[[torch.FloatTensor],
                                               torch.FloatTensor]] = F.relu,
                 bn_decay: Optional[float] = None):
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(input_dims,
                                 output_dims,
                                 kernel_size,
                                 stride=stride,
                                 padding=0,
                                 bias=use_bias)
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:

        X = self._conv2d(X)
        X = self._batch_norm(X)
        if self._activation is not None:
            X = self._activation(X)

        return X


class FullyConnected(nn.Module):
    def __init__(self,
                 input_dims: Union[int, list],
                 units: Union[int, list],
                 activations: Union[Callable[[torch.FloatTensor],
                                             torch.FloatTensor], list],
                 bn_decay: float,
                 use_bias: bool = True,
                 drop: float = None):
        super(FullyConnected, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        assert type(units) == list
        self._conv2ds = nn.ModuleList([
            Conv2D(input_dims=input_dim,
                   output_dims=num_unit,
                   kernel_size=[1, 1],
                   stride=[1, 1],
                   use_bias=use_bias,
                   activation=activation,
                   bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)
        ])

        self.drop = drop

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:

        for conv in self._conv2ds:
            if self.drop is not None:
                X = F.dropout(X, self.drop, training=self.training)
            X = conv(X)
        return X


class GLU(nn.Module):
    def __init__(self, filters, num_of_features):
        super(GLU, self).__init__()

        self.num_of_filter = filters
        self.fc0 = FullyConnected(input_dims=num_of_features,
                                  units=int(2 * filters),
                                  activations=None,
                                  bn_decay=0.1,
                                  use_bias=True)

    def forward(self, data_0):

        data = torch.squeeze(self.fc0(torch.unsqueeze(data_0, -1)), -1)

        lhs, rhs = torch.split(data, [self.num_of_filter, self.num_of_filter], dim=1)

        return lhs * torch.sigmoid(rhs)
        # return F.dropout(lhs * torch.sigmoid(rhs),0.1,self.training)


class gcn_onehop(nn.Module):
    def __init__(self, filters, num_of_features, num_of_vertices):
        super(gcn_onehop, self).__init__()

        self.num_of_filter = filters
        self.fc0 = FullyConnected(input_dims=num_of_features,
                                  units=2 * filters,
                                  activations=None,
                                  bn_decay=0.1,
                                  use_bias=True)

    def forward(self, data, adj):   
        
        data_0 = torch.einsum('vw, ncw->ncv', (adj.float(), data)).contiguous()
        
        return data_0


class Cell(nn.Module):
    def __init__(self, num_mixed_ops, filters, num_of_features,
                 num_of_vertices):
        super(Cell, self).__init__()

        self._num_mixed_ops = num_mixed_ops
        self._mixed_ops = nn.ModuleList()

        filters = int(filters[-1])
        for i in range(self._num_mixed_ops):
            if i == 0:
                self._mixed_ops += [MixedOp_first()]
            if i == 1:
                 self._mixed_ops += [MixedOp_second()]    

        self.set_mode(Mode.ONE_PATH_FIXED)

    def set_mode(self, mode):
        self._mode = mode
        for op in self._mixed_ops:
            op.set_mode(mode)          

    def arch_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].arch_parameters():
                yield p    

    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p     

    def num_weight_parameters(self):
        count = 0
        for i in range(self._num_mixed_ops):
            count += self._mixed_ops[i].num_weight_parameters()

    def forward(self, x, node_fts, adj_mats):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

             

class gc_operation(nn.Module):
    def __init__(self, filters, num_of_features, num_of_vertices, num_of_hop):
        super(gc_operation, self).__init__()
        self.N = num_of_vertices
        self.GLUs = nn.ModuleList()
        self.GCNs = nn.ModuleList()        
        for _ in range(num_of_hop):
            self.GCNs.append(gcn_onehop(int(filters[-1]), num_of_features, num_of_vertices))
            
        for _ in range(num_of_hop):
            self.GLUs.append(GLU(filters[-1], num_of_features))
        
    def forward(self, data, adj, init_index):
        gcn_outputs = []
        #r1 = data
        for i in range(len(self.GCNs)):
            data = self.GCNs[i](data, adj)
            gcn_outputs += [data]
        gcn_outputs = [
            self.GLUs[i](gcn_outputs[i][:, :, init_index*self.N:(init_index+1)*self.N]) for i in range(len(gcn_outputs))
        ]
        
        return torch.max(torch.stack(gcn_outputs, dim=0), dim=0).values            


class STCell(Cell):
    def __init__(self, num_mixed_ops, filters, num_of_features,
                 num_of_vertices, num_of_hop, use_mask, dilated_num):
        super(STCell, self).__init__(num_mixed_ops, filters, num_of_features,
                                     num_of_vertices)
        self.N = num_of_vertices
        self.use_mask = use_mask
        self.dilated_num = dilated_num
        
        self.stsgc = gc_operation(filters, num_of_features, num_of_vertices, num_of_hop)
                  
    def norm_graph(self, adj):

        adj = adj / torch.unsqueeze(adj.sum(-1), -1)

        return adj

    def forward(self, adj_first_list, adj_second_list, adj_mask_list, data):
        
        current_output = 0

        adj_outputs = [] 
        
        for i in range(self._num_mixed_ops):
            if i == 0:
                adj_list = adj_first_list
            if i == 1:
                adj_list = adj_second_list        
            
            temp = self._mixed_ops[i](adj_list)
            current_output = current_output + temp
            adj_outputs += [current_output]
        
        if self.use_mask == False:
            adj = self.norm_graph(adj_outputs[-1])
        else:
            adj = self.norm_graph(adj_outputs[-1] + adj_mask_list[0])
        
        data_long = torch.cat((data[:, :, 0:self.N], 
                               data[:, :, self.dilated_num*self.N:(self.dilated_num+1)*self.N]),2)
        out = self.stsgc(data_long, adj, 1)

        return out #bdn
        
    def weight_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].weight_parameters():
                yield p      
                
        for p in self.stsgc.parameters():
             yield p      
            

    def __repr__(self):
        out_str = []
        for i in range(self._num_mixed_ops):
            out_str += ['mixed_op: %d\n%s' % (i, self._mixed_ops[i])]

        from helper import add_indent
        out_str = 'STCell {\n%s\n}' % add_indent('\n'.join(out_str), 4)
        return out_str