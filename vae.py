"""
Basic Variational Autoencoder Implementation

references:
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    https://github.com/ttchengab/VAE/blob/main/VAE.py
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, in_dim=64, in_channels=1, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.fc_dim = hidden_dims[-1]

        # create encoder
        encoder, channels, out_dim = [], in_channels, in_dim
        for hd in hidden_dims:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=hd, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hd),
                    nn.LeakyReLU())
            )
            channels = hd
            out_dim //= 2

        if out_dim == 0:
            raise ValueError('Output dimensions of convolution layers is 0.') 

        self.channels = channels
        self.out_dim = out_dim

        # create decoder
        decoder = []
        hidden_dims.reverse()
        for hd in hidden_dims[1:]:
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, hd, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hd),
                    nn.LeakyReLU())
            )
            channels = hd

        # create model
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(out_dim*out_dim*self.fc_dim, latent_dim)
        self.fc_var = nn.Linear(out_dim*out_dim*self.fc_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, out_dim*out_dim*self.fc_dim)
        self.decoder = nn.Sequential(*decoder)
        self.dec_out = nn.Sequential(
                            nn.ConvTranspose2d(channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.Sigmoid())

    def encode(self, input):
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, self.channels, self.out_dim, self.out_dim) 
        x = self.decoder(x)
        return self.dec_out(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, z, mu, logvar

    def loss(self, output, input, mu, logvar):
        D_kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())
        loss = F.binary_cross_entropy(output, input, reduction='sum') + D_kl
        return loss / input.shape[0]

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z)
