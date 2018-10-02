

# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2, os, scipy
import matplotlib.pyplot as plt
from MyImageProcess import *
import  time


# load the p+v image, convert it to grayscale
p_imagename = os.getcwd() + '/p_009.png'
p_image = cv2.imread(p_imagename, cv2.IMREAD_GRAYSCALE)
v_imagename = os.getcwd() + '/v_009.png'
v_image = cv2.imread(v_imagename, cv2.IMREAD_GRAYSCALE)

# resize the P image and fullfill to fixed size
p_image_re = myresize(p_image, 0.65)
p_image_full = myfullimage(p_image_re, (256, 256))
# plt.imshow(p_image, 'gray')
# plt.title('the full size image p')
# plt.show(), plt.close()

# move the P image to match the V image
plotlist_index = []
plotlist = []
plotlist_index_no = 0 # index all the sum values
pv_dict = {} # save the index(0,1,..) as key, and [i,j] as value
for i in range(45, 65, 2):
	for j in range(30, 60, 2):
		p_move = myimg_move(p_image_full, i, j)
		pv_multi = v_image ^ p_move
		pv_sum = np.sum(pv_multi) # sum all the elements in PV_multiply
		pv_dict[pv_sum] = [i, j]

		plotlist.append(pv_sum)
		plotlist_index.append(plotlist_index_no)
		plotlist_index_no += 1
		print('the current index is ', np.str(i), 'and', np.str(j), '====', pv_sum)
		plt.subplot(1, 2, 2), plt.clf()
		plt.imshow(pv_multi, 'gray')
		plt.pause(0.00001)

# get the sum results and plot it
plt.figure(2)
plotlist = mymaxminnorm(plotlist)
plt.plot(plotlist_index, plotlist)
plt.show(), plt.close()

#
min_pvdict_key = np.min(sorted(pv_dict.keys()))
best_movepos = pv_dict[min_pvdict_key]
p_move = myimg_move(p_image_full, best_movepos[0], best_movepos[1])
pv_multi = v_image ^ p_move
plt.figure(), plt.imshow(pv_multi, 'gray'), plt.show(), plt.close()

###########################################################
# plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(p_image, 'gray')

plt.subplot(2,3,2)
v_ndimage= scipy.ndimage.binary_fill_holes(np.asarray(p_image).astype(int))
plt.imshow(v_ndimage, 'gray')

yihuo = np.logical_xor(p_image, v_ndimage)
plt.subplot(2,3,3)
plt.imshow(yihuo, 'gray')

# plt.figure(figsize=(10,10))
plt.subplot(2,3,4)
plt.imshow(v_image, 'gray')

plt.subplot(2,3,5)
p_ndimage = scipy.ndimage.binary_fill_holes(np.asarray(v_image).astype(int))
plt.imshow(p_ndimage, 'gray')
yihuo = np.logical_xor(v_image, p_ndimage)

plt.subplot(2,3,6)
plt.imshow(yihuo, 'gray')
plt.show(), plt.close()




###########################################################
# gray = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred =  gray #cv2.GaussianBlur(gray, (11, 11), 0)
#
# # threshold the image to reveal light regions in the
# # blurred image
# thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
# # perform a series of erosions and dilations to remove
# # any small blobs of noise from the thresholded image
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=4)
#
# # perform a connected component analysis on the thresholded
# # image, then initialize a mask to store only the "large"
# # components
# labels = measure.label(thresh, neighbors=8, background=0)
# mask = np.zeros(thresh.shape, dtype="uint8")
#
# # loop over the unique components
# for label in np.unique(labels):
# 	# if this is the background label, ignore it
# 	if label == 0:
# 		continue
#
# 	# otherwise, construct the label mask and count the
# 	# number of pixels
# 	labelMask = np.zeros(thresh.shape, dtype="uint8")
# 	labelMask[labels == label] = 255
# 	numPixels = cv2.countNonZero(labelMask)
#
# 	# if the number of pixels in the component is sufficiently
# 	# large, then add it to our mask of "large blobs"
# 	if numPixels > 300:
# 		mask = cv2.add(mask, labelMask)
#
# # find the contours in the mask, then sort them from left to right
# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# cnts = contours.sort_contours(cnts)[0]
#
# # loop over the contours
# for (i, c) in enumerate(cnts):
# 	# draw the bright spot on the image
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	((cX, cY), radius) = cv2.minEnclosingCircle(c)
# 	cv2.circle(image, (int(cX), int(cY)), int(radius),
# 		(0, 0, 255), 3)
# 	cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
# # show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)