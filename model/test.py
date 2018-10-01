#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch

import util
from evaluator import evaluateModel
from model import Model

parser = argparse.ArgumentParser(description='S2S')
parser.add_argument('--no_cuda', type=util.str2bool, nargs='?', const=True, default=True, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--no_models', type=int, default=20, help='how many models to evaluate')
parser.add_argument('--original', type=str, default='model/', help='Original path.')

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--use_emb', type=str, default='False')

parser.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
parser.add_argument('--write_n_best', type=util.str2bool, nargs='?', const=True, default=False, help='Write n-best list (n=beam_width)')

parser.add_argument('--model_path', type=str, default='model/translate.ckpt', help='Path to a specific model checkpoint.')
parser.add_argument('--model_dir', type=str, default='model/')
parser.add_argument('--model_name', type=str, default='translate.ckpt')

parser.add_argument('--valid_output', type=str, default='data/val_dials/', help='Validation Decoding output dir path')
parser.add_argument('--decode_output', type=str, default='data/test_dials/', help='Decoding output dir path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


def padSequence(tensor):
    pad_token = PAD_token
    tensor_lengths = [len(sentence) for sentence in tensor]
    longest_sent = max(tensor_lengths)
    batch_size = len(tensor)
    padded_tensor = np.ones((batch_size, longest_sent)) * pad_token

    # copy over the actual sequences
    for i, x_len in enumerate(tensor_lengths):
        sequence = tensor[i]
        padded_tensor[i, 0:x_len] = sequence[:x_len]

    padded_tensor = torch.LongTensor(padded_tensor).to(device)
    return padded_tensor, tensor_lengths


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    return '%s ' % (asMinutes(s))


def load_config(args):
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % args.model_path, 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def readDialogue(model, val_file):
    input_tensor = []; target_tensor = []; bs_tensor = []; db_tensor = []

    # Iterate over dialogue
    for idx, (usr, sys, bs, db) in enumerate(
            zip(val_file['usr'], val_file['sys'], val_file['bs'], val_file['db'])):
        tensor = [model.input_word2index(word) for word in usr.strip(' ').split(' ')] + [
            EOS_token]  # model.input_word2index(word)
        input_tensor.append(torch.tensor(tensor, dtype=torch.long, device=device))  # .view(-1, 1))

        tensor = [model.output_word2index(word) for word in sys.strip(' ').split(' ')] + [EOS_token]
        target_tensor.append(torch.tensor(tensor, dtype=torch.long, device=device))  # .view(-1, 1)

        bs_tensor.append([float(belief) for belief in bs])
        db_tensor.append([float(pointer) for pointer in db])

    input_tensor, input_lengths = padSequence(input_tensor)
    target_tensor, target_lengths = padSequence(target_tensor)
    bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

    return input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor


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
        cnt = 0
        valid_loss = 0
        for name, val_file in val_dials.items():
            if True:#'SNG' in name:
                input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor = readDialogue(model, val_file)
                output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                            db_tensor, bs_tensor)

                valid_loss += 0
                val_dials_gen[name] = output_words

                cnt += 1
                if cnt > 10:
                   break

        valid_loss /= len(val_dials)
        print('Current VALID LOSS:', valid_loss)
        with open(args.valid_output + 'val_dials_gen.json', 'w') as outfile:
            json.dump(val_dials_gen, outfile)
        evaluateModel(val_dials_gen, val_dials, mode='valid')

        # TESTING
        test_dials_gen = {}
        cnt = 0
        test_loss = 0
        for name, test_file in test_dials.items():
            if True:#'SNG' in name:
                # Iterate over dialogue
                input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor = readDialogue(model, test_file)
                output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                            db_tensor, bs_tensor)

                test_loss += 0
                test_dials_gen[name] = output_words

                cnt += 1
                if cnt > 30:
                    break

        test_loss /= len(test_dials)
        print('Current TEST LOSS:', test_loss)
        with open(args.decode_output + 'test_dials_gen.json', 'w') as outfile:
            json.dump(test_dials_gen, outfile)
        evaluateModel(test_dials_gen, test_dials, mode='test')

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
        decode(ii)
        try:
            decode(ii)
        except:
            print('cannot decode')

        args.model_path = args.original

if __name__ == '__main__':
    decodeWrapper()
