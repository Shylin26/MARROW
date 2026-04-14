import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers=nn.Sequential(
            vgg[:4],
            vgg[4:9],
            vgg[9:16],
            vgg[16:23]
        )
        for param in self.parameters():
            param.requires_grad=False
        self.eval()
    
    def forward(self,x,y):
        mean=torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std=torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x=(x-mean)/std
        y=(y-mean)/std
        loss=0
        for layer in self.layers:
            x=layer(x)
            y=layer(y)
            loss+=F.mse_loss(x,y)
        return loss

class TokenizerLoss(nn.Module):
    def __init__(self,lambda_perceptual=0.1):
        super().__init__()
        self.perceptual=PerceptualLoss()
        self.lambda_perceptual=lambda_perceptual
    def forward(self,x_recon,x_orig):
        mse_loss=F.mse_loss(x_recon,x_orig)
        perc_loss=self.perceptual(x_recon,x_orig)
        total_loss=mse_loss+self.lambda_perceptual*perc_loss
        return{
            "total_loss":total_loss,
            "mse_loss":mse_loss,
            "perceptual_loss":perc_loss
        }

