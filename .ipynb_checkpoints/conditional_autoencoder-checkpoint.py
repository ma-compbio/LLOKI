import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalAutoencoderEncoder(nn.Module):
    def __init__(self, in_channels, batch_dim, latent_dim):
        super(ConditionalAutoencoderEncoder, self).__init__()
        # The input size is now the sum of input features and batch information dimensions
        self.fc1 = nn.Linear(in_channels + batch_dim, latent_dim)

    def forward(self, x, batch_info):
        # Concatenate input with batch information along the feature axis
        x = torch.cat([x, batch_info], dim=-1)
        return F.leaky_relu(self.fc1(x))


class ConditionalAutoencoderDecoder(nn.Module):
    def __init__(self, latent_dim, batch_dim, out_channels):
        super(ConditionalAutoencoderDecoder, self).__init__()
        # The output size is conditioned on batch information
        self.fc2 = nn.Linear(latent_dim + batch_dim, out_channels)

    def forward(self, z, batch_info):
        # Concatenate latent space with batch information
        z = torch.cat([z, batch_info], dim=-1)
        return self.fc2(z)


class ConditionalAutoencoder(nn.Module):
    def __init__(self, enc_in_channels, batch_dim, latent_dim):
        super(ConditionalAutoencoder, self).__init__()
        self.encoder = ConditionalAutoencoderEncoder(enc_in_channels, batch_dim, latent_dim)
        self.decoder = ConditionalAutoencoderDecoder(latent_dim, batch_dim, enc_in_channels)

    def forward(self, x, batch_info):
        # Pass the concatenated input through the encoder and decoder
        z = self.encoder(x, batch_info)
        recon_x = self.decoder(z, batch_info)
        return recon_x

    def encode(self, x, batch_info):
        return self.encoder(x, batch_info)

    def decode(self, z, batch_info):
        return self.decoder(z, batch_info)

    def loss(self, recon_x, x):
        # Reconstruction loss
        return F.mse_loss(recon_x, x)




class SimpleAutoencoderEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(SimpleAutoencoderEncoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, latent_dim)

    def forward(self, x):
        return F.leaky_relu(self.fc1(x))


class SimpleAutoencoderDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(SimpleAutoencoderDecoder, self).__init__()
        self.fc2 = nn.Linear(latent_dim, out_channels)

    def forward(self, z):
        return self.fc2(z)


class SimpleAutoencoder(nn.Module):
    def __init__(self, enc_in_channels, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = SimpleAutoencoderEncoder(enc_in_channels, latent_dim)
        self.decoder = SimpleAutoencoderDecoder(latent_dim, enc_in_channels)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

    def encode(self, x):
        return self.encoder(x)
    
    def loss(self, recon_x, x):
        # Reconstruction loss
        return F.mse_loss(recon_x, x)





