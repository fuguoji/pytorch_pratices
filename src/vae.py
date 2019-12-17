import torch
import torch.nn as nn
import torch.nn. functional as F

class VAE(torch.nn.Module):
    def __init__(self,
                 input_size,
                 h_dim,
                 z_dim):
        super(VAE, self).__init__()
        
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_size)

    def encoder(self, x):
        h = F.relu(self.fc1(x))

        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        
        return F.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z)

        return x_reconst, mu, log_var
