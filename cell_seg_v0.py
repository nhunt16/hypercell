"""
    Hypercellularity Project
    - Preprocessing
    - Segmentation
    Latest Update: 05/25
    @author: Nicolas Hunt
    """

import skimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from skimage import data
from skimage.color import rgb2hed
from skimage.viewer import ImageViewer
from skimage.filters import threshold_mean

def Main():
    global img
    img = "glom_1.jpg"
    glom_orig = skimage.io.imread(img)
    glom_prep = prep(glom_orig)
    glom_seg = segment(glom_prep, 125)
    
    display_fig(glom_orig, glom_prep, glom_seg)


def prep(glom_orig):
    glom_hed = rgb2hed(glom_orig)[:,:,0]
    glom_smooth = skimage.filters.gaussian(glom_hed)
    
    return glom_smooth


def segment(glom_prep, thresh):
    glom_prep = skimage.morphology.opening(glom_prep)
    val = skimage.filters.threshold_otsu(glom_prep)
    glom_bin = glom_prep > val
    
    regionLabel = skimage.measure.label(glom_bin)
    regionProps = skimage.measure.regionprops(regionLabel)
    glom_m1 = skimage.morphology.opening(glom_bin)
    
    return glom_bin


def display_fig(glom_orig,glom_prep,glom_seg):
    fig, axes = plt.subplots(ncols=3, figsize=(16, 6))
    ax = axes.ravel()
    ax[0].imshow(glom_orig, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    ax[1].imshow(glom_prep, cmap=plt.cm.gray)
    ax[1].set_title('Preprocessed')
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
    Main()


