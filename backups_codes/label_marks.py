import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


def get_label_marks(input_image, showimage,show_img):
	label_image = label(input_image)

	fig, axis = plt.subplots()
	axis.imshow(showimage)

	for region in regionprops(label_image):
		print('the regions area is', region.area)
		if region.area >= 10:
			minr, minc, maxr, maxc = region.bbox
			# linewidth=0 is no rectangle; linewidth = 1or2 show the rectangle
			mark_rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
			axis.add_patch(mark_rect)

	axis.set_axis_off()
	plt.tight_layout()
	# plt.show()


def the_whole_codes(image):
	# apply threshold
	thresh = threshold_otsu(image)
	bw = closing(image > thresh, square(3))
	# remove artifacts connected to image border
	cleared = clear_border(bw)
	# label image regions
	label_image = label(cleared)
	image_label_overlay = label2rgb(label_image, image=image)
	fig, ax = plt.subplots(figsize=(10, 6))
	plt.title('the best matching result')
	ax.imshow(image_label_overlay)
	for region in regionprops(label_image):
		# take regions with large enough areas
		if region.area >= 100:
			# draw rectangle around segmented coins
			minr, minc, maxr, maxc = region.bbox
			rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0)
			ax.add_patch(rect)

	ax.set_axis_off()
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	image = data.coins()[50:-50, 50:-50]
	get_label_marks(image)