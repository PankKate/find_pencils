import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import skimage
from skimage import filters
from skimage.measure import label,regionprops
from skimage.filters import  gaussian


def processing(image):
    image = np.mean(image,2)
    image = gaussian(image, sigma=19)
    threshold = filters.threshold_otsu(image)
    image[image < threshold] = 0
    image[image > 0] = 1
    image = skimage.util.invert(image)
    labeled = label(image)
    return labeled

path = './pencils/'
files = [file for file in listdir(path)]

for file in files:

    image = plt.imread(path + file)
    labeled = processing(image)
    regions = regionprops(labeled)
    num = 0
    for region in regions:
        cur_area = region.area
        if cur_area < 50000:
            continue
        cur_perim = region.perimeter
        if (cur_area/cur_perim > 75 or cur_area/cur_perim < 45):
            continue
        cur_MAL = region.major_axis_length
        if (cur_area/cur_MAL > 140):
            continue
        cur_eccen = region.eccentricity
        if cur_eccen < 0.9:
            continue
        num+=1
 
    print("Image ", path + file," - ", num, " pencils")
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(labeled)
    plt.show()