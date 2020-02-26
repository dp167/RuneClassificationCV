import cv2
import os, os.path
import glob


def get_contours(img):
    # First make the image 1-bit and get contours
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 150, 255, 0)

    #cv2.imwrite('thresh.jpg', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # filter contours that are too large or small
    size = get_size(img)
    contours = [cc for cc in contours if contourOK(cc, size)]
    return contours

def get_size(img):
    ih, iw = img.shape[:2]
    return iw * ih

def contourOK(cc, size=1000000):
    x, y, w, h = cv2.boundingRect(cc)
    if w < 50 or h < 50: return False # too narrow or wide is bad
    area = cv2.contourArea(cc)
    return area < (size * 0.5) and area > 200

def find_boundaries(img, contours):
    # margin is the minimum distance from the edges of the image, as a fraction
    ih, iw = img.shape[:2]
    minx = iw
    miny = ih
    maxx = 0
    maxy = 0

    for cc in contours:
        x, y, w, h = cv2.boundingRect(cc)
        if x < minx: minx = x
        if y < miny: miny = y
        if x + w > maxx: maxx = x + w
        if y + h > maxy: maxy = y + h

    return (minx, miny, maxx, maxy)

def crop(img, boundaries):
    minx, miny, maxx, maxy = boundaries
    return img[miny:maxy, minx:maxx]

def process_image(fimage, fpath, t):
    img = cv2.imread(fimage)
    contours = get_contours(img)
    cv2.drawContours(img, contours, -1, (0,255,0)) # draws contours, good for debugging
    bounds = find_boundaries(img, contours)
    cropped = crop(img, bounds)
    if get_size(cropped) < 400: return # too small
    fpath+=('/')
    fpath+=(str(t))
    fpath+=('.jpg')
    print("ended")
    #print(fpath)
    cv2.imwrite(fpath, cropped)

#process_image('runes/testing/Ansuz/IMG_20200220_211502.jpg')

testing_dir= 'runes/testing/'# do somthing with this need to traverse training file
files = []
images = []
j = 0
types = ('*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.tif', '*.TIF', '*.jpg', '*.JPG', '*.JPEG', '*.jpeg')  # all should work but only .jpg was tested
for t in os.listdir(testing_dir):
    #if glob.glob(t) != []:
    testing_dir = 'runes/testing/'
    testing_dir+=t
    files.append(testing_dir)
for i in range (0,len(files)):
 for t in os.listdir(files[i]):
    j+=1
    testing_dir = files[i]
    testing_dir+='/'
    testing_dir+=t
    images.append(testing_dir)
    process_image(images[j-1], files[i], j)
