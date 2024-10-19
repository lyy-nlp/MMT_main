#!/bin/bash

python train.py  /gb/lyy/data-after-preprocess \
		--arch transformer_iwslt_de_en transformer_iwslt_de_en_nmt\
		--share-decoder-input-output-embed \
		--clip-norm 0 \
		--optimizer adam \
		--reset-optimizer \
		--lr 0.005 \
		--source-lang en \
		--target-lang de \
		--max-tokens 4096 \
		--no-progress-bar \
		--log-interval 100 \
		--weight-decay 0.1 \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.2 \
		--lr-scheduler inverse_sqrt \
		--max-update 8000 \
		--warmup-updates 2000 \
		--warmup-init-lr 1e-07 \
		--update-freq 4 \
		--adam-betas 0.9,0.98 \
		--keep-last-epochs 50 \
		--dropout 0.3 \
		--tensorboard-logdir result/en-de-bpe/bl_log1 \
		--log-format simple \
		--save-dir result \
		--eval-bleu \
		--eval-bleu-remove-bpe \
		--patience 15 \
		--fp16 \
		--encoder-embed-dim 128 \
		--encoder-ffn-embed-dim 256 \
		--decoder-embed-dim 128 \
		--decoder-ffn-embed-dim 256 \
		--encoder-layers 4 \
		--decoder-layers 4 \
		--encoder-attention-heads 4 \
		--decoder-attention-heads 4 \





