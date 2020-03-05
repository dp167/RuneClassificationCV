import cv2
import os, os.path
import random
import math
import numpy as np
import numpy as py


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 1+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def randMute(img, t, fpath):
    #randomly choose amout to rotate, dither
    image = cv2.imread(img)


    ang_rot = np.random.uniform(40)-40/2
    rows,cols,ch = image.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    image = cv2.warpAffine(image,Rot_M,(cols,rows))

    tr_x = 10*np.random.uniform()-10/2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image = cv2.warpAffine(image, Trans_M, (cols, rows))

    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+5*np.random.uniform()-5/2
    pt2 = 20+5*np.random.uniform()-5/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,shear_M,(cols,rows))

    image = augment_brightness_camera_images(image)

    fpath+="/mut"
    fpath+=str(t)
    fpath+=".jpg"
    print(fpath)
    cv2.imwrite(fpath, image)

    return image

random.seed(a=None, version=2)
saving_dir= 'runes/saving/'
save = []
mut = []
images = []
j, k = 0,0
types = ('*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.tif', '*.TIF', '*.jpg', '*.JPG', '*.JPEG', '*.jpeg')  # all should work but only .jpg was tested
for t in os.listdir(saving_dir):
    saving_dir = 'runes/saving/'
    saving_dir+=t
    save.append(saving_dir)
    mut_dir = 'runes/mutated/'
    mut_dir+=t
    save.append(saving_dir)
    mut.append(mut_dir)
#print(mut)
for i in range (0,len(mut)):
 for t in os.listdir(save[i]):
    j += 1
    saving_dir = save[i]
    saving_dir+='/'
    saving_dir+=t
    images.append(saving_dir)

    for m in range(0,7):
        k += 1
        #print(images[j-1])
        #print(mut[i])
        randMute(images[j-1], k, mut[i])