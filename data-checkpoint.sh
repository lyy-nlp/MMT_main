#!/bin/bash


python3 scripts/average_checkpoints.py \
			--inputs result \
			--num-epoch-checkpoints 12 \
			--output result/model.pt \