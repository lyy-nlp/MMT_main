# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# LICENSE file in the root directory of this source tree.
from tensorboardX import SummaryWriter
writer = SummaryWriter("./generate-fig")
import math
from typing import Any, Dict, List, NamedTuple, Optional

import torch.nn.functional as F


import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from .transformer_nmt import TransformerEncoder_nmt
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,



)
from torch import Tensor

#from fairseq.modules.transformer_layer import TransformerEncoderLayer_image

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024



@register_model("transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--img_feature_dim', type=int, default=2048,
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--pre_mix', type=bool, default=True,
                            help='if True, dont scale embeddings')
        parser.add_argument('--swap_num', type=int,
                            help='number of modalities to swap')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        src_img_features,
        # multimodel_graph,
        #src_img_features_1,
        prev_output_tokens,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        # print('src_graphs:',src_graphs.size())
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_img_features=src_img_features,
            #src_img_features_1=src_img_features_1,
            # multimodel_graph=multimodel_graph,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


@register_model("transformer_align")
class TransformerAlignModel(TransformerModel):
    """
    See "Jointly Learning to Align and Translate with Transformer
    Models" (Garg et al., EMNLP 2019).
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        self.full_context_alignment = args.full_context_alignment

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerAlignModel, TransformerAlignModel).add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='D',
                            help='Number of cross attention heads per layer to supervised with alignments')
        parser.add_argument('--alignment-layer', type=int, metavar='D',
                            help='Layer number which has to be supervised. 0 corresponding to the bottommost layer.')
        parser.add_argument('--full-context-alignment', type=bool, metavar='D',
                            help='Whether or not alignment is supervised conditioned on the full target context.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_align(args)

        transformer_model = TransformerModel.build_model(args, task)
        return TransformerAlignModel(
            transformer_model.encoder, transformer_model.decoder, args
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.forward_decoder(prev_output_tokens, encoder_out)

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        attn_args = {
            "alignment_layer": self.alignment_layer,
            "alignment_heads": self.alignment_heads,
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out, **attn_args)

        if self.full_context_alignment:
            attn_args["full_context_alignment"] = self.full_context_alignment
            _, alignment_out = self.decoder(
                prev_output_tokens,
                encoder_out,
                features_only=True,
                **attn_args,
                **extra_args,
            )
            decoder_out[1]["attn"] = alignment_out["attn"]

        return decoder_out


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        # ("txt_out", Tensor),
        # ("img_out", Tensor),
        ("x_src_img_features",Tensor),
        ("x_1",Tensor),
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
    ],
)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.dropout = args.dropout
        self.Gating = GatingMechanism(args)
        self.encoder_layerdrop = args.encoder_layerdrop
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.img_feature_dim = args.img_feature_dim
        self.img_fc = Linear(self.img_feature_dim, embed_dim)
        self.embed_tokens = embed_tokens
        self.visual_fc = Linear(embed_dim * 2, 1)

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.layers1 = nn.ModuleList([])
        self.layers1.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        #self.num_layers1 = len(self.layers1)'''
        self.swap_num = getattr(args, "swap_num",5)

        #################################
        #######  image encoder  #########
        #################################
        # self.layers_img = nn.ModuleList([])
        # self.layers_img.extend(
        #     [TransformerEncoderLayer_image(args) for i in range(args.encoder_layers)]
        # )
        ## self.num_layers = len(self.layers_img)


        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.nmt_encoder = TransformerEncoder_nmt(args, dictionary, embed_tokens)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_img_features,
        #src_img_features_1,
        # multimodel_graph,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):

        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)
        # B x T x C -> T x B x C
        #x1 = x.transpose(1, 2)
        x = x.transpose(0, 1)   #l b d
        x_1 = x

        src_img_features = src_img_features.to(torch.float16)
        src_img_features = self.img_fc(src_img_features)     # b l d
        #
        '''std_dev = 0.1
        noise = torch.normal(mean=0.0, std=std_dev, size=src_img_features.shape, device=src_img_features.device)
        src_img_features_1 = src_img_features + noise
        #src_img_features1 = src_img_features.transpose(1, 2)'''

        #src_img_features_1 = src_img_features_1.to(torch.float16)
        #src_img_features_1 = self.img_fc(src_img_features_1)'''
        #src_img_features_1 = src_img_features_1.transpose(0, 1)

        src_img_features = src_img_features.transpose(0, 1)     # 49 b d
        #src_img_features_1 = torch.zeros_like(src_img_features)


        #batch
        '''batch_size = src_img_features.size(1)
        permuted_indices = torch.randperm(batch_size)
        src_img_features = src_img_features[:, permuted_indices, :]'''

        '''fc3=nn.Linear(x.size(0), 49).to('cuda')
        x1 = fc3(x1)
        x1 = x1.transpose(1, 2)
        x1 = x1.transpose(0, 1)
        cos_similarities = torch.nn.functional.cosine_similarity(src_img_features, x1, dim=0)
        x_numpy = cos_similarities.cpu().detach().numpy()
        x_numpy = np.around(x_numpy, decimals=5)
        np.save('/home/lyy/Gating2-fairseq-inter2/注意力矩阵/x_1.npy', x_numpy)'''

        '''fc3 = nn.Linear(49, x.size(0)).to('cuda')
        src_img_features1 = fc3(src_img_features1)
        src_img_features1 = src_img_features1.transpose(1, 2)
        src_img_features1 = src_img_features1.transpose(0, 1)
        for i in range(128):
            src_img_features2 = src_img_features1[:, i, :]
            x2 = x[:, i, :]
            cos_similarities = torch.nn.functional.cosine_similarity(src_img_features2, x2, dim=1)
            x_numpy = cos_similarities.cpu().detach().numpy()
        # x_numpy = np.squeeze(x_numpy, axis=-1)
            x_numpy = np.around(x_numpy, decimals=5)
        # gate_numpy = gate_numpy * 28
            file_path = f'/home/lyy/Gating2-fairseq-inter2/注意力矩阵/x1_numpy_{i}.npy'
            np.save(file_path, x_numpy)'''

        batch_len = src_lengths[0].item()
        encoder_padding_mask = src_tokens.eq(self.padding_idx)   # b l
        encoder_padding_mask_image = torch.sum(src_img_features, dim=-1).eq(0).transpose(0, 1)  #b 49

        encoder_padding_mask_1 = src_tokens.eq(self.padding_idx)  # b l
        encoder_padding_mask_image_1 = torch.sum(src_img_features, dim=-1).eq(0).transpose(0, 1)  # b 49

        encoder_states = [] if return_all_hiddens else None

        def random_mask(image_features, mask_percentage=0.2):
            shape = image_features.shape
            num_elements = torch.numel(image_features)
            num_to_mask = int(num_elements * mask_percentage)
            mask = torch.rand(shape) < (1 - mask_percentage)
            mask = mask.cuda()
            masked_features = torch.where(mask, image_features, torch.zeros_like(image_features))
            #zero_mask_indices = (mask == 0)
            #masked_features[zero_mask_indices] = torch.rand(zero_mask_indices.sum(), device=image_features.device,dtype=image_features.dtype)
            return masked_features
        noisy_src_img_features = random_mask(src_img_features, mask_percentage=0.25)

        '''src_img_features_shuffer = src_img_features[:]
        batch_size = src_img_features_shuffer.shape[1]
        shuffled_indices = torch.randperm(batch_size)
        noisy_src_img_features = src_img_features_shuffer[:, shuffled_indices, :]'''

        #mask_matrix_tmp_tmp = torch.zeros(x.size(1), src_img_features.size(0), src_img_features.size(0)).eq(0)
        for idx, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            mask_matrix_tmp = torch.zeros(x.size(1), src_img_features.size(0), src_img_features.size(0)).eq(0)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x,src_img_features,noisy_src_img_features = layer(x, src_img_features,noisy_src_img_features,encoder_padding_mask,encoder_padding_mask_image, batch_len,idx,mask_matrix_tmp)
              #  mask_matrix_tmp_tmp = torch.eq(mask_matrix, mask_matrix_tmp.cuda())
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        #x = torch.cat([x, src_img_features], dim=0)

        '''text_index=np.random.randint(batch_len,size=self.swap_num)
        for i in range(self.swap_num):
            #temp = x[text_index].clone()
            src_img_features[text_index] = x[text_index]
            noisy_src_img_features[text_index] = x[text_index]'''

            #src_img_features[text_index] = temp
        #src_img_features_1 = src_img_features
        src_img_features = torch.mean(src_img_features, dim=0, keepdim=True)  # 1*batch*dim
        t, b, c = x.shape
        src_img_features = src_img_features.expand(t, b, c)    #l b d

        noisy_src_img_features = torch.mean(noisy_src_img_features, dim=0, keepdim=True)  # 1*batch*dim
        t, b, c = x.shape
        noisy_src_img_features = noisy_src_img_features.expand(t, b, c)

        x = x + src_img_features

        '''x1 = x.transpose(0, 2)
        gate_numpy = x1.transpose(0, 1)
        gate_numpy = gate_numpy.cpu().detach().numpy()
        # gate_numpy = np.squeeze(gate_numpy, axis=-1)
        gate_numpy = np.around(gate_numpy, decimals=5)

        cos_similarities = torch.nn.functional.cosine_similarity(x, src_img_features, dim=0)
        x_numpy = cos_similarities.cpu().detach().numpy()
        # x_numpy = np.squeeze(x_numpy, axis=-1)
        x_numpy = np.around(x_numpy, decimals=5)
        # gate_numpy = gate_numpy * 28
        np.save('/home/lyy/Gating2-fairseq-inter2/注意力矩阵/x_2.npy', x_numpy)'''

        '''for i in range(128):
            x1 = x_1[:, i, :]
            src_img_features1 = src_img_features[:, i, :]
            cos_similarities = torch.nn.functional.cosine_similarity(x1, src_img_features1, dim=1)
            x_numpy = cos_similarities.cpu().detach().numpy()
        # x_numpy = np.squeeze(x_numpy, axis=-1)
            x_numpy = np.around(x_numpy, decimals=5)
        # gate_numpy = gate_numpy * 28
            file_path = f'/home/lyy/Gating2-fairseq-inter2/注意力矩阵/x2_numpy_{i}.npy'
            np.save(file_path, x_numpy)

        for i in range(128):
            x1 = x_1[:, i, :]
            src_img_features2 = noisy_src_img_features[:, i, :]
            cos_similarities = torch.nn.functional.cosine_similarity(x1, src_img_features2, dim=1)
            x_numpy3 = cos_similarities.cpu().detach().numpy()
        # x_numpy = np.squeeze(x_numpy, axis=-1)
            x_numpy3 = np.around(x_numpy3, decimals=5)
        # gate_numpy = gate_numpy * 28
            file_path = f'/home/lyy/Gating2-fairseq-inter2/注意力矩阵/x3_numpy_{i}.npy'
            np.save(file_path, x_numpy3)'''


        #nmt = TransformerEncoder_nmt(self.myargs, self.mydictionary, self.myembed_tokens)
        out_nmt = self.nmt_encoder(src_tokens, src_lengths, src_img_features)
        x_nmt = out_nmt.encoder_out
        x_src_img_features= self.Gating(x_nmt, src_img_features)   #l b d
        x_src_img_features_1= self.Gating(x_nmt, noisy_src_img_features)


        #x1 = torch.mul(x_src_img_features, src_img_features)
        #x_out = self.Gating(x, x_src_img_features)

        return EncoderOut(
            encoder_out=x, # l b d
            x_1 =x_src_img_features_1,
            x_src_img_features=x_src_img_features,
            encoder_padding_mask=encoder_padding_mask,  # b x l
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.adaptive_softmax = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra,encoder_out

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        # print("encoder_out.encoder_padding_mask:", encoder_out.encoder_padding_mask)
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[Tensor] = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_state[:encoder_out.encoder_padding_mask.size(1)],
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        #return x, {"attn": [attn], "inner_states": inner_states, "txt_out": encoder_out.txt_out, "img_out": encoder_out.img_out}

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("transformer", "transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("transformer", "transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.gating_dim = getattr(args, "gating_dim", 512)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer", "transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer_align", "transformer_align")
def transformer_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)


@register_model_architecture("transformer_align", "transformer_wmt_en_de_big_align")
def transformer_wmt_en_de_big_align(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    transformer_wmt_en_de_big(args)


######### gating  ##########
'''class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc_img = Linear(args.gating_dim * 2, 1)

    def forward(self, x, grid_img_features):


        grid_img_features = torch.mean(grid_img_features, dim=0, keepdim=True)  ## 1*batch*dim
        t, b, c = x.shape
        grid_img_features = grid_img_features.expand(t, b, c)
        merge = torch.cat([x, grid_img_features], dim=-1)

        gate = torch.sigmoid(self.fc_img(merge))  # T B C
        img_features = torch.mul(gate, x)
        return img_features'''


class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gating_dim = 128
        # self.x_linear = Linear(, 1)
        # self.x_linear = Linear(batch_len, 1) 

        self.fc_img = Linear(self.gating_dim * 2, 1)

        # self.fc_img_x = Linear(args.gating_dim, 128)

    def forward(self, x, grid_img_features):
        # x = torch.mean(x, dim=0, keepdim=True)
        # region_x_features = torch.cat([region_img_features, x.repeat(region_img_features.size(0), 1, 1)], dim=-1)
        # region_linear_x = self.fc_img(region_x_features)
        #
        # region_sigmoid_x = torch.sigmoid(region_linear_x)  # max_len * batch * 1
        # region_img_features = torch.mul(region_sigmoid_x, region_img_features)
        # return region_img_features, region_sigmoid_x

        grid_img_features = torch.mean(grid_img_features, dim=0, keepdim=True)  ## 1*batch*dim
        t, b, c = x.shape
        grid_img_features = grid_img_features.expand(t, b, c)
        merge = torch.cat([x, grid_img_features], dim=-1)


        '''merge_mean = torch.mean(merge, dim=-1)
        merge_mean = merge_mean.transpose(0, 1)
        merge_mean = merge_mean.cpu().detach().numpy()
        merge_mean = np.around(merge_mean, decimals=5)'''

        gate = torch.sigmoid(self.fc_img(merge))  # T B C
        #        gate = torch.tanh(self.fc_img(merge))
        #        gate = torch.relu(self.fc_img(merge))
        #        gate = torch.softplus(self.fc_img(merge))

        '''gate_numpy = gate.transpose(0, 1)
        gate_numpy = gate_numpy.cpu().detach().numpy()
        gate_numpy = np.squeeze(gate_numpy, axis=-1)
        gate_numpy = np.around(gate_numpy, decimals=5)
        #gate_numpy = gate_numpy * 28
        np.save('gate_array.npy', gate_numpy)'''

        x = torch.mul(gate, x)
        #x = torch.mul(gate, x) + torch.mul(1-gate, grid_img_features)
        return  x
