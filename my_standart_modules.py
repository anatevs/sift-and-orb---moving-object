# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:42:20 2021

@author: User
"""

# for showing image in pixels and with scale
from matplotlib import pyplot as plt
px_sc = 1/plt.rcParams['figure.dpi']  # pixel in inches

# img - image to show
# im_sc - scale for showing image, default = 1
# cmap - colormap as in matplotlib, default - a colormap of the img itself
def showing_img(img, im_sc = 1, cmap = plt.rcParams['image.cmap']):
    show_sc = im_sc * px_sc #scale for showing images in plt
    fig, ax = plt.subplots(figsize=(img.shape[0]*show_sc, img.shape[1]*show_sc))
    ax.imshow(img, cmap)
    
def show_gr(img, im_sc=1):
    showing_img(img, im_sc, cmap='gray')