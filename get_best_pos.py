

# import the necessary packages
import os, scipy
from MyImageProcess import *


def find_best_pos(p_imagename, v_imagename, re_scale, move_stride, move_width, move_height):
	width_start = move_width[0]
	width_end = move_width[1]
	height_start = move_height[0]
	height_end = move_height[1]

	# load the p+v image, convert it to grayscale
	p_image = cv2.imread(p_imagename, cv2.IMREAD_GRAYSCALE)
	v_image = cv2.imread(v_imagename, cv2.IMREAD_GRAYSCALE)
	# fill all holes
	p_image = scipy.ndimage.binary_fill_holes(np.asarray(p_image).astype(int))
	v_image = scipy.ndimage.binary_fill_holes(np.asarray(v_image).astype(int))
	p_image = (p_image * 255).astype(np.uint8)
	v_image = (v_image * 255).astype(np.uint8)

	# resize the P image and fullfill to fixed size
	p_image_re = myresize(p_image, re_scale)
	p_image_full = myfullimage(p_image_re, (256, 256))
	# mypltshow('p_image_full', p_image_full)

	# move the P image to match the V image
	plotlist_index = []
	pv_sumlist = []
	plotlist_index_no = 0  # index all the sum values
	pv_dict = {}  # save the index(0,1,..) as key, and [i,j] as value
	for i in range(width_start, width_end, move_stride):
		for j in range(height_start, height_end, move_stride):
			p_move = myimagemove(p_image_full, [i, j])
			pv_multi = v_image ^ p_move
			pv_sum = np.sum(pv_multi)
			pv_dict[pv_sum] = [i, j]

			pv_sumlist.append(pv_sum)
			plotlist_index.append(plotlist_index_no)
			plotlist_index_no += 1
			print('the current index is ', np.str(i), 'and', np.str(j), '====', pv_sum)
			plt.figure, plt.clf()
			plt.imshow(pv_multi, 'gray')
			plt.pause(0.00001)

	pvdict_data = []
	for index, f in enumerate(pv_dict.keys()):
		print(index, 'xxxxxxxxx', f)
		pvdict_data.append(f)
	# pvdict_data = pv_dict.keys()
	plot_min = np.min(sorted(pv_dict.keys()))
	plot_max = np.max(sorted(pv_dict.keys()))
	pv_plotdata = (pvdict_data - plot_min)/(plot_max - plot_min)

	# find the best move pos
	min_pvdict_key = np.min(sorted(pv_dict.keys()))
	best_movepos = pv_dict[min_pvdict_key]
	return v_image, p_image_full, pv_sumlist, best_movepos, pv_plotdata


if __name__ == '__main__':
	p_imagename = os.getcwd() + '/v_results/predict_009.png'
	v_imagename = os.getcwd() + '/v_results/predict_009.png'
	re_scale = 0.65
	move_width = [45, 65]
	move_height = [30, 60]
	move_stride = 2
	find_best_pos(p_imagename, v_imagename, re_scale, move_width, move_height, move_stride)
