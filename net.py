import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from typing import Union, Callable, Optional
from mode import Mode
from cell import STCell


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


class position_embedding(nn.Module):
    def __init__(self,
                  input_length,
                  num_of_vertices,
                  embedding_size,
                  temporal=True,
                  spatial=True):
        super(position_embedding, self).__init__()
        '''
        Parameters
        ----------
        data: mx.sym.var, shape is (B, T, N, C)

        input_length: int, length of time series, T

        num_of_vertices: int, N

        embedding_size: int, C

        temporal, spatial: bool, whether equip this type of embeddings

        init: mx.initializer.Initializer

        prefix: str

        Returns
        ----------
        data: output shape is (B, T, N, C)
        '''

        self.temporal_emb = None
        self.spatial_emb = None

        if temporal:

            self.temporal_emb = nn.Parameter(torch.randn(
                1, embedding_size, input_length, 1),
                                              requires_grad=True)

            nn.init.xavier_uniform_(self.temporal_emb,
                                    gain=math.sqrt(0.0003 / 6))

        if spatial:

            self.spatial_emb = nn.Parameter(torch.randn(
                1, embedding_size, 1, num_of_vertices),
                                            requires_grad=True)
            nn.init.xavier_uniform_(self.spatial_emb,
                                    gain=math.sqrt(0.0003 / 6))

    def forward(self, data):

        if self.temporal_emb is not None:

            data = data + self.temporal_emb
        if self.spatial_emb is not None:

            data = data + self.spatial_emb

        return data


class stsgcm(nn.Module):
    def __init__(self, filters, num_of_features, num_of_vertices, 
                 num_of_hop, use_mask, dilated_num):
        super(stsgcm, self).__init__()

        self.gcn_operation = STCell(2, filters, num_of_features,
                                    num_of_vertices, num_of_hop, use_mask, dilated_num)

        self.filters = filters
        self.N = num_of_vertices

        self.set_mode(Mode.ONE_PATH_FIXED)

    def forward(self, adj_first_list, adj_second_list, adj_mask, data):

        out = self.gcn_operation(adj_first_list, adj_second_list, adj_mask, data)

        return out

    def set_mode(self, mode):
        self._mode = mode

        self.gcn_operation.set_mode(mode)

    def arch_parameters(self):

        for p in self.gcn_operation.arch_parameters():
            yield p

    def weight_parameters(self):

        for p in self.gcn_operation.weight_parameters():
            yield p


class stsgcl(nn.Module):
    def __init__(self,
                 T,
                 num_of_vertices,
                 num_of_features,
                 num_of_hop,
                 filters,
                 dilated_num, 
                 use_mask=False,
                 temporal_emb=True,
                 spatial_emb=True,
                 sts_kernal_size=None,
                 num_layer=None,
                 skip_channels=None,
                 bn_decay=None,
                 dropout=None,
                 prefix=""):
        super(stsgcl, self).__init__()
        '''
        STSGCL, multiple individual STSGCMs

        Parameters
        ----------
        data: mx.sym.var, shape is (B, T, N, C)

        adj: mx.sym.var, shape is (3N, 3N)

        T: int, length of time series, T

        num_of_vertices: int, N

        num_of_features: int, C

        filters: list[int], list of C'

        activation: str, {'GLU', 'relu'}

        temporal_emb, spatial_emb: bool

        prefix: str

        Returns
        ----------
        output shape is (B, T-2, N, C')
        '''

        self.position_embedding = position_embedding(T, num_of_vertices,
                                                     num_of_features,
                                                     temporal_emb, spatial_emb)

        self.T = T
        self.dilated_num = dilated_num
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices

        self.stsgcm = nn.ModuleList()
        self.num_recurrence = self.T - dilated_num
        for _ in range(self.num_recurrence):
            self.stsgcm.append(
                stsgcm(filters, num_of_features, num_of_vertices, 
                       num_of_hop, use_mask, dilated_num))

        self.sts_kernal_size = sts_kernal_size
        
        self.filter_convs = nn.Conv2d(in_channels=num_of_features,
                                       out_channels=num_of_features,
                                       kernel_size=(2, 1), dilation=(dilated_num, 1))
        
        self.gate_convs = nn.Conv2d(in_channels=num_of_features,
                                     out_channels=num_of_features,
                                     kernel_size=(2, 1), dilation=(dilated_num, 1))  


        self.out_sts_dim = filters[-1]

        self.skip_channels = skip_channels

        self.dropout = dropout

        self.skip = nn.Conv2d(in_channels=self.num_recurrence , out_channels=12, kernel_size=(1, 1), dilation=(1, 1))
        
        self.res = nn.Conv2d(in_channels=self.T, out_channels=self.num_recurrence , kernel_size=(1, 1), dilation=(1, 1))
        
        self.bn1 = nn.BatchNorm2d(num_of_vertices)
        
        self.bn2 = nn.BatchNorm2d(num_of_vertices)

        self.set_mode(Mode.ONE_PATH_FIXED)

    def forward(self, data, adj_first_list, adj_second_list, adj_mask):

        data = self.position_embedding(data)
        data_res = torch.tanh(self.filter_convs(data)) * torch.sigmoid(self.gate_convs(data))
        need_concat = []
        for i in range(self.num_recurrence):

            data_t = data[..., i:i + self.dilated_num+1, :].contiguous().view(
                -1, self.num_of_features,
                (self.dilated_num+1) * self.num_of_vertices)

            out = self.stsgcm[i](adj_first_list, adj_second_list, adj_mask, data_t)

            need_concat.append(out)

        out_STS = torch.stack(need_concat, dim=2)
        del need_concat

        out = out_STS + data_res

        skip = self.skip(out.permute(0,2,1,3)).permute(0,2,1,3)
        
        skip = self.bn1(skip.permute(0,3,1,2)).permute(0,3,2,1)
        
        residual = self.res(data.permute(0,2,3,1)) # shape is (B, T, N, C)  TO (B, N, T, C) TO (B, C, T, N)
        
        out = self.bn2(out.permute(0,3,2,1) + residual.permute(0,2,1,3)).permute(0,3,2,1)
        
        return out, skip

    def set_mode(self, mode):
        self._mode = mode
        for op in self.stsgcm:
            op.set_mode(mode)
        # self.tcn_operation.set_mode(mode)

    def arch_parameters(self):
        for i in range(len(self.stsgcm)):
            for p in self.stsgcm[i].arch_parameters():
                yield p

    def weight_parameters(self):
        for i in range(len(self.stsgcm)):
            for p in self.stsgcm[i].weight_parameters():
                yield p
        

        for m in [
                self.position_embedding, self.filter_convs, self.gate_convs,
                self.skip, self.res, self.bn1, self.bn2
        ]:
            for p in m.parameters():
                yield p


class mask_operation(nn.Module):
    def __init__(self, sts_kernal_size, num_nodes, node_dim):
        super(mask_operation, self).__init__()
        self.nodevec1 = nn.Parameter(torch.randn(sts_kernal_size*num_nodes, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, sts_kernal_size*num_nodes), requires_grad=True)

    def forward(self):
        adj =  F.relu(torch.mm(self.nodevec1, self.nodevec2))
        return F.dropout(adj,0.3,self.training)

        
class Net(nn.Module):
    def __init__(self,
                 steps_per_day: int,
                 bn_decay: float,
                 gcn_depth,
                 num_nodes,
                 device,
                 pre_adj_first=None,
                 pre_adj_second=None,
                 num_of_hop = 2,
                 use_mask = False,
                 dropout=0,
                 node_dim=40,
                 conv_channels=32,
                 residual_channels=32,
                 skip_channels=64,
                 end_channels=128,
                 seq_length=12,
                 in_dim=2,
                 out_dim=12,
                 layers=3,
                 forcp = None):
        
        super(Net, self).__init__()
        residual_channels = 40

        self.GWN_out = False
        self.recurrent_prediction = True
        self.seq_length = seq_length

        # tcn_channels = residual_channels

        D = residual_channels

        filter_list = np.full((layers, gcn_depth), D, dtype=int)
        print('filter_list:', filter_list)
        first_layer_embedding_size = D
        self._steps_per_day = steps_per_day

        self._fully_connected_1 = FullyConnected(input_dims=[in_dim, D],
                                                 units=[D, D],
                                                 activations=[F.relu, None],
                                                 bn_decay=bn_decay)

        sts_kernal_size = int(pre_adj_first[0].shape[0] / num_nodes)

        self.sts_kernal_size = sts_kernal_size
        additional_scope = sts_kernal_size - 1

        use_STE = True

        num_of_vertices = num_nodes
        self.num_nodes = num_nodes

        input_length = seq_length
        temporal_emb = True
        spatial_emb = True
        use_mask = True

        self.first_layer_embedding_size = first_layer_embedding_size

        self.adj_first_list = pre_adj_first
        self.adj_second_list = pre_adj_second
        self.mask = mask_operation(2, num_of_vertices, 16)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=first_layer_embedding_size,
                                    kernel_size=(1, 1))
        num_of_features = first_layer_embedding_size

        self.stsgcl = nn.ModuleList()
        receptive_field = 1
        dilated_num = 1
        for idx, filters in enumerate(filter_list):

            self.stsgcl.append(
                stsgcl(input_length,
                       num_of_vertices,
                       num_of_features,
                       num_of_hop,
                       filters,
                       dilated_num,
                       use_mask = use_mask,
                       temporal_emb=temporal_emb,
                       spatial_emb=spatial_emb,
                       sts_kernal_size=sts_kernal_size,
                       num_layer=idx,
                       skip_channels=skip_channels,
                       bn_decay=bn_decay,
                       dropout=dropout))

            input_length -= dilated_num
            num_of_features = filters[-1]
            receptive_field += dilated_num
            dilated_num = dilated_num * 2
            
        self.stsgcl.append(
            stsgcl(input_length,
                   num_of_vertices,
                   num_of_features,
                   num_of_hop,
                   filters,
                   4,
                   use_mask = use_mask,
                   temporal_emb=temporal_emb,
                   spatial_emb=spatial_emb,
                   sts_kernal_size=sts_kernal_size,
                   num_layer=idx,
                   skip_channels=skip_channels,
                   bn_decay=bn_decay,
                   dropout=dropout))    

        self.input_length_forpred = 12
        self.num_of_features_forpred = num_of_features

        # if self.recurrent_prediction:
        self.end_convs = nn.ModuleList()
        for i in range(12):
            self.end_convs.append(
                nn.Sequential(
                    OrderedDict([('fc1',
                                nn.Conv2d(in_channels=self.input_length_forpred *
                                                num_of_features,
                                                out_channels=end_channels,
                                                kernel_size=(1, 1))),
                                ('relu', nn.ReLU()),
                                ('dropout',nn.Dropout(p=0)),
                                ('fc2',
                                 nn.Conv2d(in_channels=end_channels,
                                                out_channels=1,
                                                kernel_size=(1, 1)))])))

        self.receptive_field = receptive_field


        self.dropout = dropout
        self.forcp = forcp


    def forward(self,
                input,
                task_level=12,
                mode=Mode.ONE_PATH_FIXED):

        self.set_mode(mode)

        X = input[:, :2, :, :].transpose(2, 3)

        del input

        in_len = X.size(2)
        # assert in_len == self.P, 'input sequence length not equal to preset sequence length'
        if in_len < self.receptive_field:

            X = nn.functional.pad(X, (0, 0, self.receptive_field - in_len, 0))
            assert X.size(2) == self.receptive_field, 'padding error!'

        x = self.start_conv(X) #bdtn

        del X
        skip = 0
        
        adj_mask = self.mask() 
        adj_mask_list = [adj_mask]

        for i in range(len(self.stsgcl)):
            if i < self.forcp:

                x, s = checkpoint(self.stsgcl[i], x, self.adj_first_list, 
                                  self.adj_second_list, adj_mask_list)
            else:

                x, s = self.stsgcl[i](x, self.adj_first_list, 
                                  self.adj_second_list, adj_mask_list)
            skip = skip + s
            
        x = skip

        x = x.contiguous().view(
                -1, self.num_of_features_forpred * self.input_length_forpred,
                self.num_nodes, 1)
        need_concat = []
        for i in range(task_level):

            need_concat.append(self.end_convs[i](x))

        x = torch.cat(need_concat, dim=1)

        return x

    def set_mode(self, mode):
        self._mode = mode
        for op in self.stsgcl:
            op.set_mode(mode)

    def arch_parameters(self):
        for i in range(len(self.stsgcl)):
            for p in self.stsgcl[i].arch_parameters():
                yield p

    def weight_parameters(self):
        for i in range(len(self.stsgcl)):
            for p in self.stsgcl[i].weight_parameters():
                yield p

        for m in [ self.start_conv, self.mask, self.end_convs]:
            for p in m.parameters():
                yield p