"""
Created on Sat May  8 19:12:40 2021

@author: HP
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from candidate_op import BasicOp, create_op_first, create_op_second
from mode import Mode
#import candidate_op

class MixedOp_first(BasicOp):
    def __init__(self):
        super(MixedOp_first, self).__init__()

        self._candidate_op_cell = ['TG', 'SG','TG+SG', 'SG+TG']
        self._num_ops = len(self._candidate_op_cell)
        self._candidate_ops = nn.ModuleList()
        for op_name in self._candidate_op_cell:

            self._candidate_ops += [create_op_first(op_name)]

        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops),
                                              requires_grad=True)
        self.set_mode(Mode.ONE_PATH_FIXED)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None
        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            self._sample_idx = np.array([op], dtype=np.int32)
        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(
                probs, 1, replacement=True).cpu().numpy()
        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(
                probs, 2, replacement=True).cpu().numpy()
        elif mode == Mode.ALL_PATHS:
            self._sample_idx = np.arange(self._num_ops)

    def forward(self, adj_list):
        probs = F.softmax(self._candidate_alphas[self._sample_idx], dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):

            temp = self._candidate_ops[idx](adj_list)

            output = output + probs[i] * temp
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p

    def __repr__(self):

        out_str = ''
        out_str += 'mode: ' + str(self._mode) + str(self._sample_idx) + ',\n'

        probs = F.softmax(self._candidate_alphas.data, dim=0)
        for i in range(self._num_ops):
            out_str += 'op:%d, prob: %.3f, info: %s,' % (
                i, probs[i].item(), self._candidate_ops[i])
            if i + 1 < self._num_ops:
                out_str += '\n'

        from helper import add_indent
        out_str = 'mixed_op {\n%s\n}' % add_indent(out_str, 4)
        return out_str

    def render_name(self):
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        index = torch.argmax(probs).item()
        out_str = self._candidate_ops[index].type
        out_str = '%s(%.2f)' % (out_str, probs[index])
        return out_str


class MixedOp_second(BasicOp):
    def __init__(self):
        super(MixedOp_second, self).__init__()

        self._candidate_op_cell = ['TG', 'SG', 'TC']
        self._num_ops = len(self._candidate_op_cell)
        self._candidate_ops = nn.ModuleList()
        for op_name in self._candidate_op_cell:

            self._candidate_ops += [create_op_second(op_name)]

        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops),
                                              requires_grad=True)
        self.set_mode(Mode.ONE_PATH_FIXED)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None
        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            self._sample_idx = np.array([op], dtype=np.int32)
        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(
                probs, 1, replacement=True).cpu().numpy()
        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            self._sample_idx = torch.multinomial(
                probs, 2, replacement=True).cpu().numpy()
        elif mode == Mode.ALL_PATHS:
            self._sample_idx = np.arange(self._num_ops)

    def forward(self, adj_list):
        probs = F.softmax(self._candidate_alphas[self._sample_idx], dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):

            temp = self._candidate_ops[idx](adj_list)

            output = output + probs[i] * temp
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p

    def __repr__(self):

        out_str = ''
        out_str += 'mode: ' + str(self._mode) + str(self._sample_idx) + ',\n'

        probs = F.softmax(self._candidate_alphas.data, dim=0)
        for i in range(self._num_ops):
            out_str += 'op:%d, prob: %.3f, info: %s,' % (
                i, probs[i].item(), self._candidate_ops[i])
            if i + 1 < self._num_ops:
                out_str += '\n'

        from helper import add_indent
        out_str = 'mixed_op {\n%s\n}' % add_indent(out_str, 4)
        return out_str

    def render_name(self):
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        index = torch.argmax(probs).item()
        out_str = self._candidate_ops[index].type
        out_str = '%s(%.2f)' % (out_str, probs[index])
        return out_str
    