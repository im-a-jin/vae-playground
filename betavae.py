"""
Disentangling BetaVAE Implementation

references:
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
    https://github.com/matthew-liu/beta-vae/blob/master/models.py
    https://github.com/1Konny/Beta-VAE/blob/master/model.py
"""
import torch
from torch import nn
from torch.nn import functional as F

class BetaVAE(nn.Module):
    def __init__(self, in_dim=64, in_channels=1, latent_dim=20, hidden_dims=None, beta=1, gamma=1000, capacity=25, stop_iter=100):
        super(BetaVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32]

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.conv_dim = hidden_dims[-1]

        self.beta = beta
        self.gamma = gamma
        self.capacity = capacity
        self.stop_iter = stop_iter

        encoder, channels, out_dim = [], in_channels, in_dim
        for hd in hidden_dims:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels=hd, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hd),
                    nn.LeakyReLU())
            )
            channels = hd
            out_dim //= 2

        if out_dim == 0:
            raise ValueError('Output dimensions of convolution layers is 0.')

        self.channels = channels
        self.out_dim = out_dim

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
        self.enc_fc1 = nn.Linear(out_dim*out_dim*self.conv_dim, 256)
        self.enc_fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        self.dec_in = nn.Linear(latent_dim, 256)
        self.dec_fc1 = nn.Linear(256, 256)
        self.dec_fc2 = nn.Linear(256, out_dim*out_dim*self.conv_dim, 256)
        self.decoder = nn.Sequential(*decoder)
        self.dec_out = nn.Sequential(
                            nn.ConvTranspose2d(channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.Sigmoid())

    def encode(self, input):
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        x = self.enc_fc1(x)
        x = self.enc_fc2(x)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z):
        x = self.dec_in(z)
        x = self.dec_fc1(x)
        x = self.dec_fc2(x)
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

    def b_loss(self, output, input, mu, logvar):
        D_kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1), dim=0)
        return F.binary_cross_entropy(output, input) + self.beta * D_kl

    def c_loss(self, output, input, mu, logvar, iter):
        D_kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1), dim=0)
        C = torch.Tensor([self.capacity]).to(input.device)
        C = torch.clamp(iter/self.stop_iter*C, 0, self.capacity)
        return F.binary_cross_entropy(output, input) + self.gamma * (D_kl - C).abs()

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z)
