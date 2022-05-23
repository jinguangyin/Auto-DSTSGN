"""
Created on Sat May  8 15:18:42 2021

@author: HP
"""

import torch.nn as nn

class BasicOp(nn.Module):
    def __init__(self, **kwargs):
        super(BasicOp, self).__init__()

    def forward(self, inputs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        cfg = []
        for (key, value) in self.setting:
            cfg += [str(key) + ': ' + str(value)]
        return str(self.type) + '(' + ', '.join(cfg) + ')'

    @property
    def type(self):
        raise NotImplementedError

    @property
    def setting(self):
        raise NotImplementedError


class pos_calucution(nn.Module):
    def __init__(self, adj_id):
        super(pos_calucution, self).__init__()       
        self.adj_id = adj_id

    def forward(self, adj_list):
        adj = adj_list[self.adj_id]
        out = adj
        return out


def create_op_first(op_name):
    pre_adj_v1 = 0
    pre_adj_v2 = 1
    pre_adj_v3 = 2
    pre_adj_v4 = 3    
    name2op = {
        'TG':
        lambda: pos_calucution(pre_adj_v1),
        'SG':
        lambda: pos_calucution(pre_adj_v2),
        'TG+SG':
        lambda: pos_calucution(pre_adj_v3),
        'SG+TG':
        lambda: pos_calucution(pre_adj_v4),        
    }
    op = name2op[op_name]()
    return op

def create_op_second(op_name):
    pre_adj_v1 = 0
    pre_adj_v2 = 1
    pre_adj_v3 = 2  
    name2op = {
        'TG':
        lambda: pos_calucution(pre_adj_v1),
        'SG':
        lambda: pos_calucution(pre_adj_v2),
        'TC':
        lambda: pos_calucution(pre_adj_v3),
    }
    op = name2op[op_name]()
    return op

