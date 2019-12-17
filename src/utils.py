import torch
import os
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]

        return ids.view(batch_size, -1)

class Data(object):
    def __init__(self, filename, min_count):
        self.filename = filename
        self.min_count

    def get_word(self, min_count):
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        f = open(self.filename, 'r')
        for line in f:
            self.sentence_count += 1
            sentence = line.strip().split()
            self.sentence_length += len(sentence)
            for w in sentence:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[w] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy
