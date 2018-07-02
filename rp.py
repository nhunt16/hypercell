# !/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''

import cv2

# speed-up using multithreads
cv2.setUseOptimized(True);
cv2.setNumThreads(4);

in_img_name = 'reg_img/glom_1s.jpg'
out_img_name = str(in_img_name.split('.')[0]) + '_out.jpg'
# read image
im = cv2.imread(in_img_name)
# resize image
newHeight = 200
newWidth = int(im.shape[1] * 200 / im.shape[0])
im = cv2.resize(im, (newWidth, newHeight))

# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

# # Switch to fast but low recall Selective Search method
# ss.switchToSelectiveSearchFast()

# Switch to high recall but slow Selective Search method
ss.switchToSelectiveSearchQuality()

# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# number of region proposals to show
numShowRects = len(rects)

# create a copy of original image
imOut = im.copy()

for i, rect in enumerate(rects):
    # draw rectangle for region proposal till numShowRects
    if (i < numShowRects):
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    else:
        break

# show output
cv2.imshow("Output", imOut)
cv2.imwrite(out_img_name, imOut)

# show output
cv2.imshow("Output", imOut)
cv2.imwrite(out_img_name, imOut)


# close image show window
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass