import torch
import torch.nn as nn
class FSQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        # This basis is what lets us turn [1, 2, 3] into a single unique integer
        basis = torch.tensor([1.0] + [levels[i] for i in range(len(levels)-1)], dtype=torch.float32).cumprod(0)
        self.register_buffer("basis_buf", basis)

    def bound(self, z):
        half = (self.levels - 1) / 2
        return torch.tanh(z) * half

    def quantize(self, z):
        z_bounded = self.bound(z)
        zhat = torch.round(z_bounded)
        return z_bounded + (zhat - z_bounded).detach()

    def codes_to_indices(self, codes):
        half = (self.levels - 1) / 2
        shifted = codes + half
        return (shifted * self.basis_buf).sum(dim=-1).long()

    def forward(self, z):
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        return codes, indices 


class Encoder(nn.Module):
    def __init__(self,in_channels=3,latent_dim=10):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),



        )
        self.project=nn.Conv2d(512,latent_dim,kernel_size=1)
    
    def forward(self,x):
        h=self.net(x)
        z=self.project(h)
        B,C,H,W=z.shape
        z=z.permute(0,2,3,1).reshape(B,H*W,C)
        return z

class Decoder(nn.Module):
    def __init__(self,out_channels=3,latent_dim=10):
        super().__init__()
        self.project=nn.Conv2d(latent_dim,512,kernel_size=1)
        self.net=nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,out_channels,kernel_size=4,stride=2,padding=1),
            nn.Sigmoid()
        )
    def forward(self,z):
        B,N,C=z.shape
        H,W=5,8
        z=z.reshape(B,H,W,C).permute(0,3,1,2)
        h=self.project(z)
        return self.net(h)

class VisualTokenizer(nn.Module):
    def __init__(self, levels=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5]):
        super().__init__()
        self.encoder = Encoder(latent_dim=len(levels))
        self.fsq = FSQ(levels)
        self.decoder = Decoder(latent_dim=len(levels))

    def forward(self, x):
        z = self.encoder(x)
        codes, indices = self.fsq(z) 
        x_recon = self.decoder(codes)
        return x_recon, codes, indices 


