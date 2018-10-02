from MyImageProcess import *
from label_marks import *


def detect_abnormal_regions(p_originalimage, p_image, v_originalimage, v_image,
                            best_movepos, re_scale):

	############################################
	# move p image to reasonable pos
	p_move = myimg_move(p_image, best_movepos)
	p_originalimage_re = myresize(p_originalimage, re_scale)
	p_originalimage_full = myfullimage(p_originalimage_re, (256, 256))
	p_originalmove = myimg_move(p_originalimage_full, best_movepos)
	p_yihuo = np.logical_xor(p_move, p_originalmove)
	v_yihuo = np.logical_xor(v_image, v_originalimage)

	############################################
	# 对异或图像进行腐蚀+膨胀操作，去除多余杂质
	p_yihuo = (p_yihuo * 255).astype(np.uint8)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	eroded = cv2.erode(p_yihuo, kernel)
	p_yihuo_dilated = cv2.dilate(eroded, kernel)

	############################################
	# 将可疑违禁物目标区域合并
	pv_add = np.logical_or(p_yihuo_dilated, v_yihuo)
	# get_label_marks(pv_add, v_orginalimage, show_img)
	p_yihuo_dilated_label_image = label(p_yihuo_dilated)
	p_dilated_regionprops = regionprops(p_yihuo_dilated_label_image)
	pv_label_image = label(pv_add)
	pv_regionprops = regionprops(pv_label_image)

	p_dilated_region = []
	pv_region = []
	for (p_dilated_region_tmp, pv_region_tmp) in zip(p_dilated_regionprops, pv_regionprops):
		p_dilated_region.append(p_dilated_region_tmp.area)
		pv_region.append(pv_region_tmp.area)

	same_region = [same_region for same_region in p_dilated_region if same_region in pv_region]

	fig, axis = plt.subplots()
	axis.imshow(show_img, cmap='gray')
	for region in pv_regionprops:
		if region.area in same_region:
			minr, minc, maxr, maxc = region.bbox
			minr = minr * show_height_scale
			minc = minc * show_width_scale
			maxr = maxr * show_height_scale * 1.02
			maxc = maxc * show_width_scale * 1.01
			mark_rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red',
			                               linewidth=2)
			axis.add_patch(mark_rect)
	axis.set_axis_off()
	plt.tight_layout()
	plt.draw(), plt.close()

	return same_region


if __name__ == '__main__':
	detect_abnormal_regions()
