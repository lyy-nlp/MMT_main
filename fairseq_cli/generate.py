#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys
import numpy as np
import torch.nn.functional as F
import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

from difflib import SequenceMatcher



def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw) '

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w',encoding='utf-8', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)

def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        #encoder = TransformerEncoder()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)

            attention_list = []
            for sublist in hypos:
                first_dict = sublist[0]
                attention = first_dict['attention']
                attention = attention.cpu()
                attention_np = np.array(attention)
                attention_np_rounded = np.round(attention_np, 3)
                attention_list.append(attention_np_rounded)
            save_path = '/home/lyy/Gating2-fairseq-inter2/attention'
            for i, matrix in enumerate(attention_list):
                filename = os.path.join(save_path, f'matrix_{i}.npy')
                np.save(filename, matrix)

            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)
            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                    #target_tokens_gpu = utils.strip_pad(sample['target'][i, :], tgt_dict.pad())
                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

                max_similarity = 0
                most_similar_sentence = None

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    #Embedding
                    '''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    hypo_tokens_gpu = hypo_tokens.to(device)
                    encoder = model.encoder.embed_tokens
                    embed_dim = encoder.embedding_dim
                    embed_scale =  math.sqrt(embed_dim)
                    x = embed_scale * encoder(target_tokens_gpu)
                    x = F.dropout(x, p=0.3, training=False)
                    y = embed_scale * encoder(hypo_tokens_gpu)
                    y = F.dropout(y, p=0.3, training=False)'''


                    #Cosine
                    '''def cosine_similarity_between_indices(indices1, indices2, vocab_size):
                        # 
                        vec1 = torch.zeros(vocab_size)
                        vec2 = torch.zeros(vocab_size)
                        vec1[indices1] = 1
                        vec2[indices2] = 1
                        similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
                        return similarity.item()'''

                    #（Longest Common Subsequence, LCS） 13:51.40
                    '''def lcs(X, Y):
                        m = len(X)
                        n = len(Y)
                        dp = [[0] * (n + 1) for _ in range(m + 1)]

                        for i in range(m + 1):
                            for j in range(n + 1):
                                if i == 0 or j == 0:
                                    dp[i][j] = 0
                                elif X[i - 1] == Y[j - 1]:
                                    dp[i][j] = dp[i - 1][j - 1] + 1
                                else:
                                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

                        return dp[m][n]

                    def lcs_similarity(sentence1, sentence2):
                        #words1 = sentence1.split()
                        #words2 = sentence2.split()
                        #lcs_length = lcs(words1, words2)
                        #max_len = max(len(words1), len(words2))
                        lcs_length = lcs(sentence1, sentence2)
                        max_len = max(len(sentence1), len(sentence2))
                        return lcs_length / max_len  # '''


                    #（Sequence Alignment） 16:51.05
                    '''def sequence_alignment_similarity(sentence1, sentence2):
                        return SequenceMatcher(None, sentence1, sentence2).ratio()'''

                    #（Word Overlap）
                    '''def word_overlap_similarity(sentence1, sentence2):
                        words1 = set(sentence1.split())
                        words2 = set(sentence2.split())
                        overlap = len(words1 & words2)
                        return overlap / min(len(words1), len(words2))'''


                    #Jaccard
                    #hypo_str
                    '''def jaccard_similarity(sentence1, sentence2):
                        set1 = set(sentence1.split())
                        set2 = set(sentence2.split())
                        intersection = set1.intersection(set2)
                        union = set1.union(set2)
                        jaccard_sim = len(intersection) / len(union)
                        return jaccard_sim'''

                    #hypo_token
                    '''def jaccard_similarity(tokens1, tokens2):
                        set1 = set(tokens1.tolist())
                        set2 = set(tokens2.tolist())
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        return intersection / union if union != 0 else 0.0'''


                    '''current_similarity =jaccard_similarity(target_tokens, hypo_tokens)
                    if current_similarity > max_similarity:
                        max_similarity = current_similarity
                        most_similar_sentence = hypo_str
                        most_hypo = hypo
                        most_hypo_tokens = hypo_tokens'''
                    '''else:
                        most_hypo = hypo
                        most_similar_sentence = hypo_str
                        most_hypo_tokens = hypo_tokens'''


                    if not args.quiet:
                        score = hypo['score']/ math.log(2)  # convert to base 2
                        print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                # convert from base e to base 2
                                hypo['positional_scores'].div_(math.log(2)).tolist(),
                            ))
                        ), file=output_file)

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                            ), file=output_file)

                        if args.print_step:
                            print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                        if getattr(args, 'retain_iter_history', False):
                            for step, h in enumerate(hypo['history']):
                                _, h_str, _ = utils.post_process_prediction(
                                    hypo_tokens=h['tokens'].int().cpu(),
                                    src_str=src_str,
                                    alignment=None,
                                    align_dict=None,
                                    tgt_dict=tgt_dict,
                                    remove_bpe=None,
                                )
                                print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

                '''if not args.quiet:
                    score = most_hypo['score']/ math.log(2)  # convert to base 2  
                    print('H-{}\t{}\t{}'.format(sample_id, score, most_similar_sentence), file=output_file)   
                    print('P-{}\t{}'.format(  
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                                # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)

                    if args.print_alignment:
                        print('A-{}\t{}'.format(  
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args.print_step: 
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    if getattr(args, 'retain_iter_history', False):  
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                    # Score only the top hypothesis
                if has_target : 
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)  
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, most_similar_sentence)
                    else:
                        scorer.add(target_tokens, most_hypo_tokens)'''

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        logger.info('Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
        print(scorer.result_string())
    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
