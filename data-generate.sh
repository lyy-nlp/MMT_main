#!/bin/bash

python3 generate.py  /gb/lyy/data-after-preprocess \
				--path results/model.pt \
				--source-lang en --target-lang de \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path results \
				--remove-bpe \
#				--fp16 \
#				--nbest 5 \