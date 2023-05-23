import torch
import torch.nn as nn
import textarch
import numpy as np


architecture_file = 'architecture.txt'  # Replace this with the path to your text file

class BTCVAE(nn.Module):
    def __init__(self, architecture_file, alpha=1.0, beta=4.0):
        super(BTCVAE, self).__init__()

        encoder_arch, decoder_arch, latent_dim = textarch.parse_architecture(architecture_file)

        self.encoder = self.build_layers(encoder_arch)
        self.decoder = self.build_layers(decoder_arch)
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        
        print(f"fuckin encoder {self.encoder}")

        
    def build_layers(self, arch):
        layers = []
        for layer_type, units, activation in arch:
            if layer_type == "input":
                prev_units = units
                continue
            if layer_type == "dropout":
                layers.append(nn.Dropout(units))
                continue
            if layer_type == "dense":
                layers.append(nn.Linear(prev_units, units))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "lrelu":
                    layers.append(nn.LeakyReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "mish":
                    layers.append(nn.Mish())
            prev_units = units
        
        return nn.Sequential(*layers)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_log_var = self.encoder(x)
        print("got here")
        mu, log_var = torch.chunk(mu_log_var, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def kl_divergence(self, mu, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        log_qz = torch.logsumexp(0.5 * (-logvar - (mu.pow(2) + logvar.exp())), dim=1)
        log_pz = torch.logsumexp(-0.5 * mu.pow(2), dim=1)
        tc_loss = self.alpha * (log_qz - log_pz).sum()
        return kl_div, tc_loss

    def loss_function(self, x, x_recon, mu, logvar, check, edges, nunique):
        recon_loss = self.loss_split(x, x_recon, check, edges, nunique)
        kl_loss, tc_loss = self.kl_divergence(mu, logvar)
        loss = recon_loss + self.beta * kl_loss + tc_loss
        return loss
        
    def loss_split(self, x, x_recon, check, edges, nunique):

        print(f"input arrays check {check.dtype} edges {edges.dtype} nunique {nunique.dtype}")
        mse_t_idxs = np.where(check == 0)[0]
        hot_t_idxs = np.where(check == 1)[0]
        
        ohe_loss = 0
        for i in hot_t_idxs:
            ohe_recon = x_recon[edges[i] : edges[i] + nunique[i]]
            ohe_x = x[edges[i] : edges[i] + nunique[i]]
            ohe_loss += (nn.functional.binary_cross_entropy(ohe_recon, ohe_x, reduction='mean'))

        mse_x = x[edges[mse_t_idxs]]
        mse_xrecon = x_recon[edges[mse_t_idxs]]
        mse_loss = nn.functional.mse_loss(mse_xrecon, mse_x, reduction='mean')

        # Example of taking a simple sum
        total_loss = ohe_loss + mse_loss

        # Alternatively, take a weighted sum
        # total_loss = 0.5 * ohe_loss + 0.5 * mse_loss
        return total_loss





# btcvae = BTCVAE(architecture_file)
# print(btcvae)
