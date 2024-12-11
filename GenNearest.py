
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske, GEN_TYPE=1):           
        """ generator of image from skeleton """
        
        min = Skeleton.distance(self.videoSkeletonTarget.ske[0], ske)
        ind = 0
        for i, ske_t in enumerate(self.videoSkeletonTarget.ske):
            ske_distance = Skeleton.distance(ske_t, ske)
            if ske_distance < min:
                min = ske_distance
                ind = i

        img = cv2.imread("data/" + self.videoSkeletonTarget.im[ind])   # reads an image in the BGR format
     

        return np.array(img)




