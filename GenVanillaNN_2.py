import numpy as np
import cv2
import os
import pickle
import sys
import math
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is', device)

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None, preprocess_mode='SKEIMG'):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        self.preprocess_mode = preprocess_mode
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()

    ######
    def __getitem__(self, idx): 
        ske = self.videoSke.ske[idx]
        image = Image.open(self.videoSke.imagePath(idx))

        if self.preprocess_mode == 'default':
            ske = self.preprocessSkeleton(ske)
            if self.target_transform:
                image = self.target_transform(image)
            return ske, image
        else:
            in_img = np.zeros_like(np.array(image))
  
            Skeleton.draw_reduced(ske.__array__(reduced=self.ske_reduced), in_img)
        
            if self.target_transform:
                image = self.target_transform(image)
     
            im_tensor = torch.from_numpy(in_img.transpose(2, 0, 1)).float()
        
            im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=(128,128), mode='bilinear', align_corners=False)
            return im_tensor.squeeze(0), image
    ######

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class GenNNSkeToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            # Start from 1x1 and gradually increase the spatial dimensions layer by layer

            nn.ConvTranspose2d(self.input_dim, 32, kernel_size=4, stride=1, padding=0),  # Output: 4x4
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 8x8
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 15x15
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 30x30
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),  # Output: 59x59
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),  # Slight increase
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1),   # Output: 63x63
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0),   # Minor refinement to 64x64
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(),

            # Final layer to produce the output with 3 channels (RGB)
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=0),    # Output: 64x64
            nn.Tanh()  # Output values in the range [-1, 1]

        )
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        
        return img.view(img.size(0), 3, 64, 64)




class GenNNSkeImToImage(nn.Module):
    """ class that Generate a new image from from THE IMAGE OF the new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
 
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            # contracting block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # expanding block
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), 
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),  
            nn.Tanh() 
        )
        
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        
        img = self.model(z)
        return img.view(img.size(0), 3, 64, 64)






class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64

        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

        if optSkeOrImage==1:

            self.netG = GenNNSkeToImage().to(device)
            src_transform = None
            self.filename = 'data/Dance/DanceGenVanillaFromSke_16112024.pth'
            self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform, preprocess_mode='default')
            self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)
        else:
            self.netG = GenNNSkeImToImage().to(device)
            src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeImg_13112024.pth'
            self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
            self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)


        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.netG.parameters(), lr=0.001, momentum=0.9)
        

        for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
               
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.netG(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
                    torch.save(self.netG, self.filename)
        torch.save(self.netG, self.filename)
        print('Finished Training')


    
    def generate(self, ske, GEN_TYPE):
        """ Generate an image from skeleton data.
            Or from a skeleton image
        """
   
        # If mode is 'draw', initialize an empty image and draw the skeleton
        if (GEN_TYPE == 3 or GEN_TYPE==4) :
            in_img = np.zeros((128, 128, 3))
            Skeleton.draw_reduced(ske.__array__(reduced=True), in_img)
            
            # Convert to tensor and transpose for (C, H, W) format
            im_tensor = torch.from_numpy(in_img.transpose(2, 0, 1)).float()
            ske_t_batch = im_tensor.unsqueeze(0).to(device)  # Create batch

        # If mode is 'default', preprocess the skeleton directly
        else:
       
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0).to(device)  # Create batch

        # Forward pass through the generator
        normalized_output = self.netG(ske_t_batch)
        
        # Convert the output tensor to an image
        res = self.dataset.tensor2image(normalized_output[0].cpu())  # Get image 0 from the batch
        return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command-line arguments")

    parser.add_argument('--optSkeOrImage', type=int, default=1, help="Generate from Ske coordinates or Ske image")
    parser.add_argument('--n_epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--filename', type=str, default="data/taichi1.mp4", help="Path to the input video file")
    parser.add_argument('--force', action='store_true', help="Force option (set to True to overwrite existing data)")

    args = parser.parse_args()

    force = False

    n_epoch = args.n_epochs  # 200
    train = 1 #False

    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", args.filename)
    print("GenVanillaNN: Filename=", args.filename)

    targetVideoSke = VideoSkeleton(args.filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=args.optSkeOrImage)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i], args.optSkeOrImage )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)