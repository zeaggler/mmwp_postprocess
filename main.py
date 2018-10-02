
from get_best_pos import *
import cv2
import time
import scipy.io as scio

############################################
process_image_name = '010.png'
re_scale = 0.79
move_width_start = -100
move_width_end = 128
move_height_start = -100
move_height_end = 128
move_stride = 2
save_flag = False
move_width = [move_width_start, move_width_end]
move_height = [move_height_start, move_height_end]

############################################
v_imagename = os.getcwd() + '/v_results/predict_' + process_image_name
v_originalimage = cv2.imread(v_imagename, cv2.IMREAD_GRAYSCALE)
p_imagename = os.getcwd() + '/p_results/predict_' + process_image_name
p_originalimage = cv2.imread(p_imagename, cv2.IMREAD_GRAYSCALE)

v_showname = os.getcwd() + '/v_show/' + process_image_name
show_img = cv2.imread(v_showname, 0)
show_txtimg = cv2.cvtColor(cv2.imread(v_showname, 0), cv2.COLOR_GRAY2BGR)

# compute the scale to locate detective region in visual image
[show_height, show_width] = np.shape(show_img)
show_height_scale = show_height/256
show_width_scale = show_width/256

v_image, p_image, pv_sumlist, best_movepos, pv_plotdata = \
	find_best_pos(p_imagename, v_imagename, re_scale, move_stride, move_width, move_height)


############################################
start_time = time.time()
# move p image to reasonable pos
p_move = myimagemove(p_image, best_movepos)
p_originalimage_re = myresize(p_originalimage, re_scale)
p_originalimage_full = myfullimage(p_originalimage_re, (256, 256))
p_originalmove = myimagemove(p_originalimage_full, best_movepos)
p_yihuo = np.logical_xor(p_move, p_originalmove)*255
v_yihuo = np.logical_xor(v_image, v_originalimage)*255
plt.subplot(1,3,1), plt.imshow(v_originalimage)
plt.subplot(1,3,2), plt.imshow(v_image)
plt.subplot(1,3,3), plt.imshow(v_yihuo)
plt.show()
############################################


############################################
# 对异或图像进行腐蚀+膨胀操作，去除多余杂质
p_yihuo = (p_yihuo * 255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded = cv2.erode(p_yihuo, kernel)
p_yihuo_dilated = cv2.dilate(eroded, kernel)*255
############################################


############################################
# generate the best dic of the hidden object position to be saved
pv_add = np.logical_or(p_yihuo_dilated, v_yihuo)*255
tmp_dict = {'image': v_showname[v_showname.rindex('/')+1:]}
tmp_dict.update({'movepara': [re_scale, move_stride, best_movepos[0], best_movepos[1]]})
tmp_dict.update(mydetectdic(p_yihuo_dilated/255, pv_add/255, v_yihuo/255, [show_height_scale, show_width_scale]))

time_elapsed = (time.time() - start_time)*1000
print('The code run {:.0f}ms'.format(time_elapsed % 60))
############################################


############################################
mytxt2deldic('savepara.txt', process_image_name)  # first del the current image line
with open('savepara.txt', 'a') as f:  # second write the current image to txt
	f.write(str(tmp_dict))
	f.write('\n')
	f.close()
# third reload txt to read the detection pos to show
txt2dic = mytxt2dic('savepara.txt', process_image_name)
for rec_index in range(len(txt2dic)-2):
	rec = txt2dic[rec_index]
	print('rec is', rec)
	cv2.rectangle(show_txtimg, (rec[0], rec[1]), (rec[2], rec[3]), (0, 0, 255), 2)
cv2.imshow('showdetect', show_txtimg)
cv2.waitKey(0)
############################################


############################################
plt.subplot(2, 4, 1), plt.imshow(v_originalimage, 'gray'), plt.title('v_originalimage')
plt.subplot(2, 4, 2), plt.imshow(p_originalimage, 'gray'), plt.title('p_orginalimage')
plt.subplot(2, 4, 3), plt.imshow(v_image, 'gray'), plt.title('v_image')
plt.subplot(2, 4, 4), plt.imshow(p_move, 'gray'), plt.title('p_image')
plt.subplot(2, 4, 5), plt.imshow(v_yihuo, 'gray'), plt.title('v_yihuo')
plt.subplot(2, 4, 6), plt.imshow(p_yihuo_dilated, 'gray'), plt.title('p_yihuo_dilated')
plt.subplot(2, 4, 7), plt.imshow(pv_add, 'gray'), plt.title('pv_add')
plt.subplot(2, 4, 8), plt.plot(pv_plotdata), plt.title('pv_plotdata')
plt.show()
plt.close('all')

############################################
if save_flag:
    savefilename = process_image_name[:process_image_name.rindex('.')]
    makefilename = os.getcwd() + '/savefiles/' + savefilename
    isExists = os.path.exists(makefilename)
    if not isExists:
    	os.makedirs(makefilename)
    imagesavepath = makefilename + '/v_image.png'
    cv2.imwrite(imagesavepath, v_image)
    imagesavepath = makefilename + '/p_move.png'
    cv2.imwrite(imagesavepath, p_move)
    imagesavepath = makefilename + '/v_yihuo.png'
    cv2.imwrite(imagesavepath, v_yihuo.astype(np.uint8))
    imagesavepath = makefilename + '/p_yihuo_dilated.png'
    cv2.imwrite(imagesavepath, p_yihuo_dilated.astype(np.uint8))
    imagesavepath = makefilename + '/pv_add.png'
    cv2.imwrite(imagesavepath, pv_add.astype(np.uint8))
    cv2.waitKey(5)
    imagesavepath = makefilename + '/detect.png'
    cv2.imwrite(imagesavepath, show_txtimg)
    cv2.waitKey(5)
    plotdataNew = makefilename + '/plotdata.mat'
    scio.savemat(plotdataNew, {'data': pv_plotdata})
    print('....................................the code is over....................................')

cv2.destroyAllWindows()
