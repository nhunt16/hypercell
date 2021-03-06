"""
Hypercellularity Project
	- Preprocessing
	- Segmentation
Latest Update: 05/25
@author: Nicolas Hunt
"""

import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from skimage import data
from skimage.color import rgb2hed
from skimage.viewer import ImageViewer
from skimage.filters import threshold_mean
from pathlib import Path
from datetime import datetime


class CellSeg:
    def __init__(self, img_dir):
        file_path = Path(__file__)
        self.in_dir = '{}/{}'.format(str(file_path.parent), img_dir)
        self.prep_out_dir = 'prep-' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.seg_out_dir = 'seg-' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.in_images = os.listdir(self.in_dir)
        self.preprocess = self.prep()
        self.prep_images = os.listdir(self.prep_out_dir)
        self.thresh = 125
        self.segment = self.seg(self.prep_out_dir, self.thresh)

    def prep(self, glom_orig=None):
        try:
            os.stat(self.prep_out_dir)
        except:
            os.mkdir(self.prep_out_dir)

        if glom_orig == None:
            for i in self.in_images:
                # print(i)
                glom_orig = skimage.io.imread(self.in_dir + '/' + i)
                glom_hed = rgb2hed(glom_orig)[:, :, 0]
                glom_smooth = skimage.filters.gaussian(glom_hed)

                hed_file = '{}/{}_h.jpg'.format(self.prep_out_dir, i.split(".")[0])
                scipy.misc.imsave(hed_file, glom_hed)

        else:
            glom_hed = rgb2hed(glom_orig)[:, :, 0]
            glom_smooth = skimage.filters.gaussian(glom_hed)

            return glom_smooth

    def seg(self, glom_prep_dir=None, thresh=None, glom_prep=None):
        try:
            os.stat(self.seg_out_dir)
        except:
            os.mkdir(self.seg_out_dir)
        if glom_prep_dir is not None and thresh is not None:
            for i in self.prep_images:
                glom_hed = skimage.io.imread(glom_prep_dir + '/' + i)
                val = skimage.filters.threshold_otsu(glom_hed)
                glom_bin = glom_hed > val

                regionLabel = skimage.measure.label(glom_bin)
                regionProps = skimage.measure.regionprops(regionLabel)
                glom_m1 = skimage.morphology.opening(glom_bin)

                bin_file = '{}/{}_b.jpg'.format(self.seg_out_dir, i.split("_h")[0])
                print(glom_bin)
                scipy.misc.imsave(bin_file, glom_bin)
        else:
            glom_prep = skimage.morphology.opening(glom_prep)
            val = skimage.filters.threshold_otsu(glom_prep)
            glom_bin = glom_prep > val

            regionLabel = skimage.measure.label(glom_bin)
            regionProps = skimage.measure.regionprops(regionLabel)
            glom_m1 = skimage.morphology.opening(glom_bin)

            return glom_bin

    def display_fig(self, glom_orig, glom_prep, glom_seg):
        fig, axes = plt.subplots(ncols=3, figsize=(16, 6))
        ax = axes.ravel()
        ax[0].imshow(glom_orig, cmap=plt.cm.gray)
        ax[0].set_title('Original image')
        ax[1].imshow(glom_prep, cmap=plt.cm.gray)
        ax[1].set_title('HED to RGB')
        ax[2].imshow(glom_seg, cmap=plt.cm.gray, interpolation='nearest')
        ax[2].set_title('Segmented')
        for a in ax:
            a.axis('off')

        # Display
        plt.show()
        # Save plot
        plot_file = img.split(".")[0] + "_plot_v2" + ".jpg"
        fig.savefig(plot_file, bbox_inches='tight')


if __name__ == '__main__':
    # global img
    # img = "glom_1.jpg"
    # glom_orig = skimage.io.imread(img)
    # glom_prep = prep(glom_orig)
    # glom_seg = seg(glom_prep, 125)
    #
    # display_fig(glom_orig, glom_prep, glom_seg)
    seg = CellSeg('glom_he')
    print(seg.in_dir)
