import numpy as np
import cv2
import os
import pickle
import sys
import argparse
from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest

from GenVanillaNN_2 import *
from GenGAN import *


class DanceDemo:
    """ class that run a demo of the dance.
        The animation/posture from self.source is applied to character define self.target using self.gen
    """
    def __init__(self, filename_src, typeOfGen=2):
        self.target = VideoSkeleton( "data/taichi1.mp4" )
        self.source = VideoReader(filename_src)
        if typeOfGen==1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen==2:         # VanillaNN
            print("Generator: GenSimpleNN")
            self.generator = GenVanillaNN( self.target, loadFromFile=True, optSkeOrImage=1)
        elif typeOfGen==3:         # VanillaNN
            print("Generator: GenSimpleNN")
            self.generator = GenVanillaNN( self.target, loadFromFile=True, optSkeOrImage=2)
        elif typeOfGen==4:         # GAN
            print("Generator: GenSimpleNN")
            self.generator = GenGAN( self.target, loadFromFile=True)
        else:
            print("DanceDemo: typeOfGen error!!!")


    def draw(self, GEN_TYPE):
        ske = Skeleton()
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)  # (B, G, R)
        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()
            if i%5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
                if isSke:
                    ske.draw(image_src)

                    image_tgt = self.generator.generate(ske, GEN_TYPE)            # GENERATOR !!!
                    if GEN_TYPE != 1:
                        image_tgt = image_tgt*255
                    image_tgt = cv2.resize(image_tgt, (128, 128))
                else:
                    image_tgt = image_err
                image_combined = combineTwoImages(image_src, image_tgt)
                image_combined = cv2.resize(image_combined, (512, 256))
                cv2.imshow('Image', image_combined)
                cv2.imwrite(f'data/images/output_image{i}.png', image_tgt)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    self.source.readNFrames( 100 )
        cv2.destroyAllWindows()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command-line arguments")

    parser.add_argument('--GEN_TYPE', type=int, default=1, help="The model to use")

    args = parser.parse_args()



    # NEAREST = 1
    # VANILLA_NN_SKE = 2
    # VANILLA_NN_Image = 3
    # GAN = 4
    GEN_TYPE =args.GEN_TYPE
    ddemo = DanceDemo("data/taichi2_full.mp4", GEN_TYPE)

    ddemo.draw(GEN_TYPE)
