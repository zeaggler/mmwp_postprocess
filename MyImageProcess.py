
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
# from label_marks import *

def myimagemove(input_image, best_movepos):
	h_move = best_movepos[0]
	v_move = best_movepos[1]
	height, width = input_image.shape
	matrix_move = np.float32([[1, 0, h_move], [0, 1, v_move]])
	dst = cv2.warpAffine(input_image, matrix_move, (height, width))
	return dst


def mymaxminnorm(input_image):
	mymax = max(input_image)
	mymin = min(input_image)
	output = (input_image - mymin) / (mymax - mymin)
	return output


def myresize(input_image, times):
	height, width = input_image.shape
	size = (int(width*times), int(height*times))
	output = cv2.resize(input_image, size, interpolation=cv2.INTER_AREA)
	return output


def myfullimage(input_image, target_size):
	output = np.zeros(target_size)
	output[:input_image.shape[0], :input_image.shape[0]] = input_image
	output = output.astype(np.uint8)
	return output


def mypltshow(title, input_image):
	plt.imshow(input_image, 'gray')
	plt.title(title)
	plt.show()
	plt.close()


# only return the row line specified by imagename
def mytxt2dic(txtname, com_imagename):
	txt2dic = {}
	with open(txtname, 'r') as f:
		for f_index in f.readlines():
			txt2dic_tmp = eval(f_index)
			if txt2dic_tmp['image'] == com_imagename:
				txt2dic.update(txt2dic_tmp)
	return txt2dic


# delete the line specified by imagename and return the txt2deldic
def mytxt2deldic(txtname, com_imagename):
	with open(txtname, 'r') as f:
		txt2deldic = f.readlines()
		f.close()
	for index_no, index_data in enumerate(txt2deldic):
		if com_imagename in index_data:
			del txt2deldic[index_no: index_no + 1]
	with open(txtname, 'w') as f:
		f.writelines(txt2deldic)


def mydetectdic(p_yihuo_dilated, pv_add, v_yihuo, show_scale):
	p_dilated_regionprops = regionprops(label(p_yihuo_dilated))
	pv_regionprops = regionprops(label(pv_add))
	pv_area = []
	pd_area = []
	for region in pv_regionprops:
		pv_area.append(region.area)
	for region in p_dilated_regionprops:
		pd_area.append(region.area)
	# find the same region in two target images
	same_region = list(set(pv_area).intersection(pd_area))
	[show_height_scale, show_width_scale] = show_scale
	tmp_dict = {}
	para_no = 0
	for region in pv_regionprops:
		v_label_flag = True
		if region.area in same_region:
			region_coords = region.coords
			for coordinates in region_coords:
				if v_yihuo[coordinates[0], coordinates[1]] > 0:
					v_label_flag = False
					break
			if v_label_flag:
				minr, minc, maxr, maxc = region.bbox
				minr = int(minr * show_height_scale)
				minc = int(minc * show_width_scale)
				maxr = int(maxr * show_height_scale * 1.02)
				maxc = int(maxc * show_width_scale * 1.01)
				tmp_dict_para = [minc, minr, maxc, maxr]
				tmp_dict.update({para_no: tmp_dict_para})
				para_no += 1
	return tmp_dict
