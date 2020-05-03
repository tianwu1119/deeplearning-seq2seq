# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from fairseq import utils
import h5py

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, input, target):

        x = input
        y = target

        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost#, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        print(x.size(), y.size())
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        print(x_col.size(), y_lin.size())
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        print(C.size())
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

@register_criterion('d2gpo_label_smoothed_cross_entropy')
class D2GPoLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        # load the pretrained weight
        assert args.d2gpo_weight_path is not None
        assert args.d2gpo_vocab_path is not None
        
        fw = h5py.File(args.d2gpo_weight_path, 'r')
        lw = fw['weights']
        self.d2gpo_weights = np.array(lw)
        fw.close()

        with open(args.d2gpo_vocab_path, 'r', encoding='utf-8') as fin:
            data = fin.readlines()
        self.d2gpo_vocab = [line.strip() for line in data if len(line.strip())>0]

        assert len(task.target_dictionary) == self.d2gpo_weights.shape[0] and self.d2gpo_weights.shape[0] == self.d2gpo_weights.shape[1] and self.d2gpo_weights.shape[0] == len(self.d2gpo_vocab)

        # check the vocabulary
        for widx in range(len(task.target_dictionary)):
            assert task.target_dictionary.symbols[widx] == self.d2gpo_vocab[widx]

        self.d2gpo_alpha = args.d2gpo_alpha
        self.d2gpo_temperature = args.d2gpo_temperature

        self.d2gpo_criterion_ = args.d2gpo_criterion
        if args.d2gpo_criterion == 'wassdistance':
            self.d2gpo_criterion = SinkhornDistance(eps=0.00001, max_iter=10, reduction='none')
        else:
            self.d2gpo_criterion = nn.KLDivLoss(reduction='none')

        self.d2gpo_post_softmax = args.d2gpo_post_softmax

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--d2gpo-alpha', default=0.1, type=float,
                            help='d2gpo alpha')
        parser.add_argument('--d2gpo-temperature', default=2.0, type=float,
                            help='d2gpo temperature')
        parser.add_argument('--d2gpo-weight-path', type=str,
                            help='d2gpo weight path')
        parser.add_argument('--d2gpo-vocab-path', type=str,
                            help='d2gpo vocabulary path')

        parser.add_argument('--d2gpo-post-softmax', action="store_true",
                            help='d2gpo post softmax')

        parser.add_argument('--d2gpo-criterion', type=str,
                            help='d2gpo criterion')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, kd_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'kd_loss': utils.item(kd_loss.data) if reduce else kd_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        #probs = lprobs.exp()

        target_weights = torch.from_numpy(self.d2gpo_weights[target.squeeze(-1).cpu()]).type_as(lprobs)

        T = self.d2gpo_temperature

        # kd_loss = self.d2gpo_criterion(input=F.log_softmax(probs / T, dim=-1), 
        #                            target=F.softmax(target_weights / T, dim=-1))
        if self.d2gpo_post_softmax:
            if self.d2gpo_criterion_ == 'wassdistance':
                kd_loss = self.d2gpo_criterion(input=lprobs.exp(), 
                                    target=F.softmax(target_weights / T, dim=-1))
            else:
                kd_loss = self.d2gpo_criterion(input=lprobs, 
                                    target=F.softmax(target_weights / T, dim=-1))
        else:
            if self.d2gpo_criterion_ == 'wassdistance':
                kd_loss = self.d2gpo_criterion(input=lprobs.exp(), 
                                    target=target_weights)
            else:
                kd_loss = self.d2gpo_criterion(input=lprobs, 
                                target=target_weights)
        
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)

        non_pad_mask = target.ne(self.padding_idx)
        kd_loss = kd_loss.sum(dim=-1, keepdim=True)
        kd_loss = kd_loss[non_pad_mask]
        kd_loss = kd_loss.sum()

        # loss = loss * (1. - self.d2gpo_alpha) + kd_loss * self.d2gpo_alpha * T * T

        loss = loss * (1. - self.d2gpo_alpha) + kd_loss * self.d2gpo_alpha * T * T

        return loss, nll_loss, kd_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'kd_loss': sum(log.get('kd_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
