
import numpy as np
import cv2
import os
import pickle
import sys
import math
import argparse
import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN_2 import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is', device)

class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
   
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),  # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)),  # 1x1
            nn.Sigmoid()
        )
        print(self.model)

    def forward(self, input):
       
        return self.model(input)
    

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """

    
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeImToImage().to(device)
        self.netD = Discriminator().to(device)
        self.lambda_gp = 10.0  # Weight for gradient penalty
 
        self.filename = 'data/Dance/DanceGenGAN_15112024.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)
        self.disc_dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)  # Smaller batch size for discriminator
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename).to(device)
          


    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        
        if fake_samples.size(0) > real_samples.size(0):
            idx = torch.randperm(fake_samples.size(0))[:real_samples.size(0)]
            fake_samples = fake_samples[idx]

        # Random interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolates = D(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        # Compute L2 norm of gradients and apply penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, n_epochs=20):
        gen_criterion = nn.BCELoss()
        disc_criterion = nn.BCELoss()
        gen_criterion_mse = nn.MSELoss()  
        gen_optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
        disc_optimizer = torch.optim.Adam(self.netD.parameters(), lr=0.00005, betas=(0.5, 0.999))


        for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times

            running_disc_loss = 0.0
            running_gen_loss = 0.0
            #for i, (inputs, labels ) in enumerate(self.dataloader):
            for i, ((disc_inputs, disc_labels), (inputs, labels)) in enumerate(zip(self.disc_dataloader, self.dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                disc_inputs, disc_labels = disc_inputs.to(device), disc_labels.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs_fake = self.netG(inputs)
                
                # zero the parameter gradients
                if i % 2 == 0:
                    disc_optimizer.zero_grad()

                    # forward + backward + optimize
                    
                    disc_real_pred = self.netD(disc_labels).view(-1)
                    disc_real_loss = disc_criterion(disc_real_pred, torch.full_like(disc_real_pred, 0.9, device=device))

                    disc_fake_pred = self.netD(outputs_fake.detach()).view(-1)
                    disc_fake_loss = disc_criterion(disc_fake_pred, torch.full_like(disc_fake_pred, 0.1, device=device))

                    gradient_penalty = self.lambda_gp * self.compute_gradient_penalty(self.netD, disc_labels, outputs_fake.detach())
                    disc_loss = ((disc_fake_loss + disc_real_loss) * 0.5) + gradient_penalty
                        
                    # Update gradients
                    disc_loss.backward(retain_graph=True)
                    # Update optimizer
                    disc_optimizer.step()
                    running_disc_loss += disc_loss.item()

                ## Update Generator ##
                gen_optimizer.zero_grad()
                disc_fake_pred = self.netD(outputs_fake).view(-1)
               
                gen_loss = gen_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred).to(device)) + 100 * gen_criterion_mse(outputs_fake, labels)
                gen_loss.backward()
                gen_optimizer.step()

                # print statistics
                running_gen_loss += gen_loss.item()
            
            if (epoch + 1) % 50 == 0:
                avg_disc_loss = running_disc_loss / len(self.dataloader)
                avg_gen_loss = running_gen_loss / len(self.dataloader)
                print(f'Epoch [{epoch + 1}/{n_epochs}], Discriminator Loss: {avg_disc_loss:.3f}, Generator Loss: {avg_gen_loss:.3f}')
            
            # Save model every `save_interval` epochs
            if (epoch + 1) % 500 == 0:
                # Create a file name with the epoch number
                model_filename = f"{self.filename}_epoch_{epoch + 1}.pth"
                torch.save(self.netG.state_dict(), model_filename)
                print(f"Model saved to {model_filename}")
        torch.save(self.netG, self.filename)
        print('Finished Training')


    def generate(self, ske, GEN_TYPE=4):           # TP-TODO
        """ generator of image from skeleton """
    
        in_img = np.zeros((128,128,3))
        
        Skeleton.draw_reduced(ske.__array__(reduced=True), in_img)
    
        im_tensor = torch.from_numpy(in_img.transpose(2, 0, 1)).float()
        
        ske_t_batch = im_tensor.unsqueeze(0).to(device)        # make a batch

        self.netG.eval()
        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch)
            res = self.dataset.tensor2image(normalized_output[0].cpu())       # get image 0 from the batch
        return res

              


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="command-line arguments")

    parser.add_argument('--n_epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--filename', type=str, default="data/taichi1.mp4", help="Path to the input video file")
    parser.add_argument('--force', action='store_true', help="Force option (set to True to overwrite existing data)")

    args = parser.parse_args()


    n_epochs = args.n_epochs
    force = False
    train = 1 #False

    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", args.filename)

    targetVideoSke = VideoSkeleton(args.filename)

    #if False:
    if train:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(n_epochs)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        print("i==", i)
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

