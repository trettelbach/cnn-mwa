import numpy as np
import scipy.ndimage
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import random

pictures = []

grid_size = 4000
grid_size_ext = int(3.5*grid_size)+1
print(grid_size_ext)

workspace = "/group/mwasci/trettelbach/test_rfi_generation/*.fits"

for filename in glob.iglob(workspace):
    # open the "good" mwa fits images
    hdulist = fits.open(filename)
    header = hdulist[0].header
    myimages csd= hdulist[0].data[0][0]
    hdulist.close()
    pictures.append(myimages)

    # creat empty image of largest necessary size
    grid = np.ones(shape=(grid_size_ext, grid_size_ext))

    # define size for output image
    grid_final = np.ones(shape=(grid_size, grid_size))

    # set frequency, amplitude, and angle at with rfi 'occurs'
    freq = random.uniform(0.6, 0.3)
    amp = random.uniform(0.8*abs(myimages.min()), 1.2*abs(myimages.min()))
    angle = random.randint(0, 179)

    # create random noise for rfi lines
    noise = np.random.normal(0, 0.15*abs(myimages.min()), myimages.shape)

    # start with adding waves in x direction
    for i in range(0, grid_size_ext):
        grid[i, :] = amp*np.sin(freq*i)

    # rotate image
    grid_rot = scipy.ndimage.rotate(grid, angle)

    # find center to re-place output image 
    cx = int((grid_size_ext-1)/2.)

    # set cutout pixels for final grid
    grid_final[:, :] = grid_rot[cx-int(grid_size/2.):cx+int(grid_size/2.), cx-int(grid_size/2.):cx+int(grid_size/2.)]

    # overlay original 'good' image with rfi lines, keep max value of pixel location
    output = np.where(grid_final > myimages, grid_final, myimages)

    # add noise grid
    signal = output + noise

    # write result to new .fits file
    hdu = fits.PrimaryHDU(signal)
    hdu.writeto("/group/mwasci/trettelbach/test_rfi_generation/rfied/" + filename[-40:-5] + "_rfi.fits")

