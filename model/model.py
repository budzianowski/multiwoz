from __future__ import division, print_function, unicode_literals

import json
import math
import operator
import os
import random
from io import open
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from functools import reduce
import policy

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


# Shawn beam search decoding
class BeamSearchNode(object):
    def __init__(self, h, prevNode, wordid, logp, leng):
        self.h = h
        self.prevNode = prevNode
        self.wordid = wordid
        self.logp = logp
        self.leng = leng

    def eval(self, repeatPenalty, tokenReward, scoreTable, alpha=1.0):
        reward = 0
        alpha = 1.0

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(1.0)
        hh_b[l // 4:l // 2].data.fill_(1.0)


def init_gru(gru, gain=1):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i+gru.hidden_size],gain=gain)


def whatCellType(input_size, hidden_size, cell_type, dropout_rate):
    if cell_type == 'rnn':
        cell = nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'gru':
        cell = nn.GRU(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'lstm':
        cell = nn.LSTM(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell
    elif cell_type == 'bigru':
        cell = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'bilstm':
        cell = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell


class EncoderRNN(nn.Module):
    def __init__(self, input_size,  embedding_size, hidden_size, cell_type, depth, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.n_layers = depth
        self.dropout = dropout
        self.bidirectional = False
        if 'bi' in cell_type:
            self.bidirectional = True
        padding_idx = 3
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.rnn = whatCellType(embedding_size, hidden_size,
                    cell_type, dropout_rate=self.dropout)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        input_lens = np.asarray(input_lens)
        input_seqs = input_seqs.transpose(0,1)
        #batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx))
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()

        if isinstance(hidden, tuple):
            hidden = list(hidden)
            hidden[0] = hidden[0].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
            hidden[1] = hidden[1].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
            hidden = tuple(hidden)
        else:
            hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()

        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)

        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)  # [T,B,H] -> [B,T,H]
        attn_energies = self.score(H,encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        cat = torch.cat([hidden, encoder_outputs], 2)
        energy = torch.tanh(self.attn(cat)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class SeqAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout_p=0.1, max_length=30):
        super(SeqAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.output_size = output_size
        self.n_layers = 1
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size + hidden_size, hidden_size, cell_type, dropout_rate=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        self.score = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, embedding_size)

        # attention
        self.method = 'concat'
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, input, hidden, encoder_outputs):
        if isinstance(hidden, tuple):
            h_t = hidden[0]
        else:
            h_t = hidden
        encoder_outputs = encoder_outputs.transpose(0, 1)
        embedded = self.embedding(input)  # .view(1, 1, -1)
        # embedded = F.dropout(embedded, self.dropout_p)

        # SCORE 3
        max_len = encoder_outputs.size(1)
        h_t = h_t.transpose(0, 1)  # [1,B,D] -> [B,1,D]
        h_t = h_t.repeat(1, max_len, 1)  # [B,1,D]  -> [B,T,D]
        energy = self.attn(torch.cat((h_t, encoder_outputs), 2))  # [B,T,2D] -> [B,T,D]
        energy = torch.tanh(energy)
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        attn_weights = F.softmax(energy, dim=2)  # [B,1,T]

        # getting context
        context = torch.bmm(attn_weights, encoder_outputs)  # [B,1,H]

        # context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) #[B,1,H]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2)
        rnn_input = rnn_input.transpose(0, 1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden  # , attn_weights


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        padding_idx = 3
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx
                                      )
        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size, hidden_size, cell_type, dropout_rate=dropout)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1,B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        output = embedded
        #output = F.relu(embedded)

        output, hidden = self.rnn(output, hidden)

        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)

        return output, hidden


class Model(nn.Module):
    def __init__(self, args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index):
        super(Model, self).__init__()
        self.args = args
        self.max_len = args.max_len

        self.output_lang_index2word = output_lang_index2word
        self.input_lang_index2word = input_lang_index2word

        self.output_lang_word2index = output_lang_word2index
        self.input_lang_word2index = input_lang_word2index

        self.hid_size_enc = args.hid_size_enc
        self.hid_size_dec = args.hid_size_dec
        self.hid_size_pol = args.hid_size_pol

        self.emb_size = args.emb_size
        self.db_size = args.db_size
        self.bs_size = args.bs_size
        self.cell_type = args.cell_type
        if 'bi' in self.cell_type:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.depth = args.depth
        self.use_attn = args.use_attn
        self.attn_type = args.attention_type

        self.dropout = args.dropout
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.teacher_forcing_ratio = args.teacher_ratio
        self.vocab_size = args.vocab_size
        self.epsln = 10E-5


        torch.manual_seed(args.seed)
        self.build_model()
        self.getCount()
        try:
            assert self.args.beam_width > 0
            self.beam_search = True
        except:
            self.beam_search = False

        self.global_step = 0

    def cuda_(self, var):
        return var.cuda() if self.args.cuda else var

    def build_model(self):
        self.encoder = EncoderRNN(len(self.input_lang_index2word), self.emb_size, self.hid_size_enc,
                                  self.cell_type, self.depth, self.dropout).to(self.device)

        self.policy = policy.DefaultPolicy(self.hid_size_pol, self.hid_size_enc, self.db_size, self.bs_size).to(self.device)

        if self.use_attn:
            if self.attn_type == 'bahdanau':
                self.decoder = SeqAttnDecoderRNN(self.emb_size, self.hid_size_dec, len(self.output_lang_index2word), self.cell_type, self.dropout, self.max_len).to(self.device)
        else:
            self.decoder = DecoderRNN(self.emb_size, self.hid_size_dec, len(self.output_lang_index2word), self.cell_type, self.dropout).to(self.device)

        if self.args.mode == 'train':
            self.gen_criterion = nn.NLLLoss(ignore_index=3, size_average=True)  # logsoftmax is done in decoder part
            self.setOptimizers()

    def train(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor, dial_name=None):
        proba, _, decoded_sent = self.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor)

        proba = proba.view(-1, self.vocab_size)
        self.gen_loss = self.gen_criterion(proba, target_tensor.view(-1))

        self.loss = self.gen_loss
        self.loss.backward()
        grad = self.clipGradients()
        self.optimizer.step()
        self.optimizer.zero_grad()

        #self.printGrad()
        return self.loss.item(), 0, grad

    def setOptimizers(self):
        self.optimizer_policy = None
        if self.args.optim == 'sgd':
            self.optimizer = optim.SGD(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adadelta':
            self.optimizer = optim.Adadelta(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adam':
            self.optimizer = optim.Adam(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)

    def forward(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor):
        """Given the user sentence, user belief state and database pointer,
        encode the sentence, decide what policy vector construct and
        feed it as the first hiddent state to the decoder."""
        target_length = target_tensor.size(1)

        # for fixed encoding this is zero so it does not contribute
        batch_size, seq_len = input_tensor.size()

        # ENCODER
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)

        # POLICY
        decoder_hidden = self.policy(encoder_hidden, db_tensor, bs_tensor)

        # GENERATOR
        # Teacher forcing: Feed the target as the next input
        _, target_len = target_tensor.size()
        decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=self.device)

        proba = torch.zeros(batch_size, target_length, self.vocab_size)  # [B,T,V]

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            use_teacher_forcing = True if random.random() < self.args.teacher_ratio else False
            if use_teacher_forcing:
                decoder_input = target_tensor[:, t].view(-1, 1)  # [B,1] Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

            proba[:, t, :] = decoder_output

        decoded_sent = None

        return proba, None, decoded_sent

    def predict(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor):
        with torch.no_grad():
            # ENCODER
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)

            # POLICY
            decoder_hidden = self.policy(encoder_hidden, db_tensor, bs_tensor)

            # GENERATION
            decoded_words = self.decode(target_tensor, decoder_hidden, encoder_outputs)

        return decoded_words, 0

    def decode(self, target_tensor, decoder_hidden, encoder_outputs):
        decoder_hiddens = decoder_hidden

        if self.beam_search:  # wenqiang style - sequicity
            decoded_sentences = []
            for idx in range(target_tensor.size(0)):
                if isinstance(decoder_hiddens, tuple):  # LSTM case
                    decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
                else:
                    decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
                encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

                # Beam start
                self.topk = 1
                endnodes = []  # stored end nodes
                number_required = min((self.topk + 1), self.topk - len(endnodes))
                decoder_input = torch.LongTensor([[SOS_token]], device=self.device)

                # starting node hidden vector, prevNode, wordid, logp, leng,
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()  # start the queue
                nodes.put((-node.eval(None, None, None, None),
                           node))

                # start beam search
                qsize = 1
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break

                    # fetch the best node
                    score, n = nodes.get()
                    decoder_input = n.wordid
                    decoder_hidden = n.h

                    if n.wordid.item() == EOS_token and n.prevNode != None:  # its not empty
                        endnodes.append((score, n))
                        # if reach maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)

                    log_prob, indexes = torch.topk(decoder_output, self.args.beam_width)
                    nextnodes = []

                    for new_k in range(self.args.beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval(None, None, None, None)
                        nextnodes.append((score, node))

                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))

                    # increase qsize
                    qsize += len(nextnodes)

                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for n in range(self.topk)]

                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_words = utterances[0]
                decoded_sentence = [self.output_index2word(str(ind.item())) for ind in decoded_words]
                #print(decoded_sentence)
                decoded_sentences.append(' '.join(decoded_sentence[1:-1]))

            return decoded_sentences

        else:  # GREEDY DECODING
            decoded_sentences = self.greedy_decode(decoder_hidden, encoder_outputs, target_tensor)
            return decoded_sentences

    def greedy_decode(self, decoder_hidden, encoder_outputs, target_tensor):
        decoded_sentences = []
        batch_size, seq_len = target_tensor.size()
        decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=self.device)

        decoded_words = torch.zeros((batch_size, self.max_len))
        for t in range(self.max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # get candidates
            topi = topi.view(-1)

            decoded_words[:, t] = topi
            decoder_input = topi.detach().view(-1, 1)

        for sentence in decoded_words:
            sent = []
            for ind in sentence:
                if self.output_index2word(str(int(ind.item()))) == self.output_index2word(str(EOS_token)):
                    break
                sent.append(self.output_index2word(str(int(ind.item()))))
            decoded_sentences.append(' '.join(sent))

        return decoded_sentences

    def clipGradients(self):
        grad = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
        return grad

    def saveModel(self, iter):
        print('Saving parameters..')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.encoder.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.enc')
        torch.save(self.policy.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.pol')
        torch.save(self.decoder.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.dec')

        with open(self.model_dir + self.model_name + '.config', 'w') as f:
            f.write(str(json.dumps(vars(self.args), ensure_ascii=False, indent=4)))

    def loadModel(self, iter=0):
        print('Loading parameters of iter %s ' % iter)
        self.encoder.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.enc'))
        self.policy.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.pol'))
        self.decoder.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.dec'))

    def input_index2word(self, index):
        if index in self.input_lang_index2word:
            return self.input_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def output_index2word(self, index):
        if index in self.output_lang_index2word: 
            return self.output_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def input_word2index(self, index):
        if index in self.input_lang_word2index:
            return self.input_lang_word2index[index]
        else:
            return 2

    def output_word2index(self, index):
        if index in self.output_lang_word2index:
            return self.output_lang_word2index[index]
        else:
            return 2

    def getCount(self):
        learnable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_cnt = sum([reduce((lambda x, y: x * y), param.shape) for param in learnable_parameters])
        print('Model has', param_cnt, ' parameters.')

    def printGrad(self):
        learnable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        for idx, param in enumerate(learnable_parameters):
            print(param.grad, param.shape)
