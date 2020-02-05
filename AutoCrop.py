import cv2
import numpy as np
import sys
import os
import Pandas

def get_points(img,size)
#corner detection return points for corner of rune



def crop(img,points)
#use found points to crop image and save cropped image with proper label 




def main(thresh, crop, filename):
    img = cv2.imread(filename)
    size = np.shape(img)
    
    corners = get_points(img,size)
    
    crop(img,corners)
    
