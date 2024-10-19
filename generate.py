#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq_cli.generate import cli_main
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cli_main()
