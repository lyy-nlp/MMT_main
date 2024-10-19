# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.kl_loss_coeff1 = args.kl_loss_coeff1
        self.kl_loss_coeff2 = args.kl_loss_coeff2

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--kl_loss_coeff1', default=10000, type=float, metavar='D',
                            help='the loss coefficient ')
        parser.add_argument('--kl_loss_coeff2', default=1, type=float, metavar='D',
                            help='the loss coefficient ')
        # fmt: on
    def kl_loss(self, x, y, epsilon=1e-7):
        kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1) + epsilon, reduction='mean')
        return kl

    def adversarial_loss(self,x_src_img_features, x_src_img_features_1):
        loss = torch.mean(torch.abs(x_src_img_features - x_src_img_features_1))
        return loss

    def calculate_L(self,x, y, T):
        #L_final = torch.Tensor.cuda()
        '''numerator = torch.exp(x/T)
        denominator = torch.exp(y/T)
        L = -torch.log(numerator/denominator+numerator)
        L = L.mean()'''
        #L_final.torch.from_numpy(L).cuda()
        l_pos =  torch.exp(x/T)
        l_neg = torch.exp(y/T)
        #logits = torch.cat([l_pos,l_neg],dim=0)   #2l b d
        #lable = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
        loss = -torch.log( l_pos / torch.add(l_pos,l_neg ) )
        loss = loss.mean()
        return loss



    def forward(self, model, model_nmt,sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        net_output_nmt = model_nmt(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, net_output_nmt,sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, net_output_nmt,sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        #kl_loss = 0
        encoder_out = net_output[2]
        encoder_out_nmt = net_output_nmt[2]
        x1 = encoder_out.encoder_out   #49 b d
        x2 = encoder_out_nmt.encoder_out      #l  b d
        x3=encoder_out.x_src_img_features    #  l b d
        x4= encoder_out.x_1
        #kl_loss2 = torch.mean((x2 - x3) ** 2)
        #kl_loss1 = self.kl_loss(x2, x3)
        #x1 = x1[:x2.size(0), :, :]
        kl_loss1 = torch.mean((x1 - x2) ** 2)
        kl_loss2 = self.calculate_L(x3, x4,1)
        #kl_loss2 = self.kl_loss(x1, x2)
        print('loss1:', loss)
        loss = loss + self.kl_loss_coeff1 * kl_loss1+ self.kl_loss_coeff2 * kl_loss2
        print('loss:', loss)
        print('kl_loss1:', kl_loss1)
        print('kl_loss2:', kl_loss2)
        print('loss:', loss)
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
