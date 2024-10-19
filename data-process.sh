#!/bin/bash

python3 preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref text/train.bpe \
  --validpref text/valid.bpe \
  --testpref text/test.bpe \
  --nwordssrc 17200 \
  --nwordstgt 9800 \
  --workers 12 \
  --destdir data-after-preprocess\

  
