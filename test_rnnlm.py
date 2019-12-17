import torch
import numpy as np
import argparse
import time

from src.utils import Dictionary, Corpus
from src.rnnlm import RNNLMModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        default='./data/Penn_Treebank/train.txt')
    parser.add_argument('--output_file',
                        default='./data/Penn_Treebank/sample.txt')
    parser.add_argument('--embed_size', 
                        default=128, 
                        type=int)
    parser.add_argument('--hidden_size',
                        default=1024,
                        type=int)
    parser.add_argument('--num_layers',
                        default=1,
                        type=int)
    parser.add_argument('--epochs',
                        default=5,
                        type=int)
    parser.add_argument('--num_samples',
                        default=1000,
                        type=int)
    parser.add_argument('--batch_size',
                        default=20,
                        type=int)
    parser.add_argument('--seq_length',
                        default=30,
                        type=int)
    parser.add_argument('--lr',
                        default=0.002,
                        type=float)

    args = parser.parse_args()

    return args

def get_data(args):
    corpus = Corpus()
    ids = corpus.get_data(args.input_file, args.batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // args.seq_length

    return corpus, ids, vocab_size, num_batches

def train(args, ids, vocab_size, num_batches):
    model = RNNLMModel(vocab_size=vocab_size,
                       embed_size=args.embed_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_layers)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        states = (torch.zeros(args.num_layers, args.batch_size, args.hidden_size),
                  torch.zeros(args.num_layers, args.batch_size, args.hidden_size))
        for i in range(0, ids.size(1)-args.seq_length, args.seq_length):
            inputs = ids[:, i:i+args.seq_length]
            targets = ids[:, (i+1):(i+1)+args.seq_length]

            states = [state.detach() for state in states]
            outputs, states = model(inputs, states)
            loss = loss_func(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            step = (i+1) // args.seq_length
            if step % 100 == 0:
                print('Epoch {}/{}, Step {}/{}, Loss: {:.4f}'.format(epoch+1, args.epochs, step, num_batches, loss.item()))
    
    return model

def test(args, model, corpos, vocab_size):
    with open(args.output_file, 'w') as f:
        state = (torch.zeros(args.num_layers, 1, args.hidden_size),
                 torch.zeros(args.num_layers, 1, args.hidden_size))
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1)

        for i in range(agrs.num_samples):
            output, state = model(input, state)

            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            input.fill(word_id)

            word = corpus.dictionary.idx2word[word_id]
            if word == '<eos>':
                word = '\n' 
            else:
                word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled {}/{} words'.format(i+1, args.num_samples))

def main():
    args = get_args()
    corpus, ids, vocab_size, num_batches = get_data(args)
    model = train(args, ids, vocab_size, num_batches)
    test(args, model, corpus, vocab_size)

if __name__ == '__main__':
    main()
