import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SkipGram(torch.nn.Module):
    def __init__(self,
                 input_size,
                 embed_dim):
        super(SkipGram, self).__init__()
        
        self.input_size = input_size
        self.embed_dim = embed_dim

        self.u_embeddings = nn.Embedding(self.input_size, embed_dim, sparse=True)
        self.v_embeddings = nn.Embedding(self.input_size, self.embed_dim, sparse=True)
        self.init_embeddings()

    def init_embeddings(self):
        self.u_embeddings.weight.data.uniform_(-0.5/self.embed_dim, 0.5/self.embed_dim)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save_embeddings(self, id2word, filename='embeddings.txt'):
        embeddings = self.u_embeddings.weight.data.numpy()
        
        with open(filename, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.embed_dim))
            for word_id, word in id2word.items():
                emb = embeddings[word_id]
                f.write('{} {}\n'.format(word, ' '.join(str(x) for x in emb)))

def test():
    model = SkipGram(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embeddings(id2word)

if __name__ == '__main__':
    test()