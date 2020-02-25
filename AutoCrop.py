import cv2
from os.path import basename
import glob

def get_contours(img):
    # First make the image 1-bit and get contours
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 150, 255, 0)

    cv2.imwrite('thresh.jpg', thresh)
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

def process_image(fname):
    img = cv2.imread(fname)
    contours = get_contours(img)
    cv2.drawContours(img, contours, -1, (0,255,0)) # draws contours, good for debugging
    bounds = find_boundaries(img, contours)
    cropped = crop(img, bounds)
    if get_size(cropped) < 400: return # too small
    print("ended")
    cv2.imwrite('croppedimage1.png', cropped)

#process_image('runetest1.png')

testing_dir= '/testing/Ansuz/'# do somthing with this need to traverse training file
files = []
types = ('*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.tif', '*.TIF', '*.jpg', '*.JPG', '*.JPEG', '*.jpeg')  # all should work but only .jpg was tested
for t in types:
    if glob.glob(t) != []:
        files.append(glob.glob(t))
for f in files[0]:
    process_image(f)# need to edit this glob functionality to traverse training data file
