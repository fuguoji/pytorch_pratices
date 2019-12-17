import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import argparse

from src.vae import VAE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size',
                        default=784,
                        type=int)
    parser.add_argument('--h_dim',
                        default=400,
                        type=int)
    parser.add_argument('--z_dim',
                        default=20,
                        type=int)
    parser.add_argument('--epochs',
                        default=10,
                        type=int)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float)

    args = parser.parse_args()

    return args

def get_data(args):
    dataset = torchvision.datasets.MNIST(root='./mnist',
                                         train=True,
                                         transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    return data_loader

def train(args, data_loader):
    model = VAE(input_size=args.input_size,
                h_dim=args.h_dim,
                z_dim=args.z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        for i, (x, _) in enumerate(data_loader):
            x = x.view(-1, args.input_size)
            x_reconst, mu, log_var = model(x)
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch {}/{}, Step {}/{}, Loss: {:.4f}'
                      .format(epoch+1, args.epochs, i+1, len(data_loader), loss.item()))
    
    return model

def test(args, model):
    z = torch.randn(args.batch_size, args.z_dim)
    output = model.decoder(z).view(-1, 1, 28, 28)
    save_image(output, os.path.join('./mnist/MNIST/sampled/', 'sampled.png'))

def main(args):
    data_loader = get_data(args)
    model = train(args, data_loader)
    test(args, model)

if __name__ == '__main__':
    main(get_args())
