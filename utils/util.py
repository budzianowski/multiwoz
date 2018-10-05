'''
Utility functions
'''

import argparse
import cPickle as pkl
import json
import sys
import math
import time
import numpy as np
import torch

# DEFINE special tokens
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

    padded_tensor = torch.LongTensor(padded_tensor)
    return padded_tensor, tensor_lengths


def loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor):
    # Iterate over dialogue
    for idx, (usr, sys, bs, db) in enumerate(
            zip(val_file['usr'], val_file['sys'], val_file['bs'], val_file['db'])):
        tensor = [model.input_word2index(word) for word in usr.strip(' ').split(' ')] + [
            EOS_token]  # model.input_word2index(word)
        input_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1))

        tensor = [model.output_word2index(word) for word in sys.strip(' ').split(' ')] + [EOS_token]
        target_tensor.append(torch.LongTensor(tensor))  # .view(-1, 1)

        bs_tensor.append([float(belief) for belief in bs])
        db_tensor.append([float(pointer) for pointer in db])

    return input_tensor, target_tensor, bs_tensor, db_tensor


#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    return '%s ' % (asMinutes(s))