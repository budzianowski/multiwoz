#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import json
import os
import shutil
import time

import torch

from utils import util
from evaluate import MultiWozEvaluator
from model.model import Model

parser = argparse.ArgumentParser(description='S2S')
parser.add_argument('--no_cuda', type=util.str2bool, nargs='?', const=True, default=True, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--no_models', type=int, default=20, help='how many models to evaluate')
parser.add_argument('--original', type=str, default='model/model/', help='Original path.')

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--use_emb', type=str, default='False')

parser.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
parser.add_argument('--write_n_best', type=util.str2bool, nargs='?', const=True, default=False, help='Write n-best list (n=beam_width)')

parser.add_argument('--model_path', type=str, default='model/model/translate.ckpt', help='Path to a specific model checkpoint.')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--model_name', type=str, default='translate.ckpt')

parser.add_argument('--valid_output', type=str, default='model/data/val_dials/', help='Validation Decoding output dir path')
parser.add_argument('--decode_output', type=str, default='model/data/test_dials/', help='Decoding output dir path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


def load_config(args):
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % args.model_path, 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def loadModelAndData(num):
    # Load dictionaries
    with open('data/input_lang.index2word.json') as f:
        input_lang_index2word = json.load(f)
    with open('data/input_lang.word2index.json') as f:
        input_lang_word2index = json.load(f)
    with open('data/output_lang.index2word.json') as f:
        output_lang_index2word = json.load(f)
    with open('data/output_lang.word2index.json') as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    if args.load_param:
        model.loadModel(iter=num)

    # Load data
    if os.path.exists(args.decode_output):
        shutil.rmtree(args.decode_output)
        os.makedirs(args.decode_output)
    else:
        os.makedirs(args.decode_output)

    if os.path.exists(args.valid_output):
        shutil.rmtree(args.valid_output)
        os.makedirs(args.valid_output)
    else:
        os.makedirs(args.valid_output)

    # Load validation file list:
    with open('data/val_dials.json') as outfile:
        val_dials = json.load(outfile)

    # Load test file list:
    with open('data/test_dials.json') as outfile:
        test_dials = json.load(outfile)
    return model, val_dials, test_dials


def decode(num=1):
    model, val_dials, test_dials = loadModelAndData(num)
    evaluator_valid = MultiWozEvaluator("valid")
    evaluator_test = MultiWozEvaluator("test")

    start_time = time.time()
    for ii in range(2):
        if ii == 0:
            print(50 * '-' + 'GREEDY')
            model.beam_search = False
        else:
            print(50 * '-' + 'BEAM')
            model.beam_search = True

        # VALIDATION
        val_dials_gen = {}
        valid_loss = 0
        for name, val_file in val_dials.items():
            input_tensor = [];  target_tensor = [];bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)

            valid_loss += 0
            val_dials_gen[name] = output_words

        print('Current VALID LOSS:', valid_loss)
        with open(args.valid_output + 'val_dials_gen.json', 'w') as outfile:
            json.dump(val_dials_gen, outfile)
        evaluator_valid.evaluateModel(val_dials_gen, val_dials, mode='valid')

        # TESTING
        test_dials_gen = {}
        test_loss = 0
        for name, test_file in test_dials.items():
            input_tensor = [];  target_tensor = [];bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, test_file, input_tensor, target_tensor, bs_tensor, db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)
            test_loss += 0
            test_dials_gen[name] = output_words

        test_loss /= len(test_dials)
        print('Current TEST LOSS:', test_loss)
        with open(args.decode_output + 'test_dials_gen.json', 'w') as outfile:
            json.dump(test_dials_gen, outfile)
        evaluator_test.evaluateModel(test_dials_gen, test_dials, mode='test')

    print('TIME:', time.time() - start_time)


def decodeWrapper():
    # Load config file
    with open(args.model_path + '.config') as f:
        add_args = json.load(f)
        for k, v in add_args.items():
            setattr(args, k, v)

        args.mode = 'test'
        args.load_param = True
        args.dropout = 0.0
        assert args.dropout == 0.0

    # Start going through models
    args.original = args.model_path
    for ii in range(1, args.no_models + 1):
        print(70 * '-' + 'EVALUATING EPOCH %s' % ii)
        args.model_path = args.model_path + '-' + str(ii)
        try:
            decode(ii)
        except:
            print('cannot decode')

        args.model_path = args.original


if __name__ == '__main__':
    decodeWrapper()
