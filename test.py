# d={}
# while(1):
# 	key=str(input("Enter the key (int) to be added:"))
# 	value=int(input("Enter the value for the key to be added:"))
#
# 	d.update({key:value})
# 	print("Updated dictionary is:")
# 	print(d)



# dic = {}
# with open('savepara.txt', 'r') as f:
# 	fdata = f.readlines()
# 	for item in fdata:
# 		if item.isspace(): #判断是否有值
# 			continue
# 		else:
# 			item = item[item.rindex('{')+1:item.rindex('}')]
# 			block = item.split(',')
# 			for index in block:
# 				v = index.split(':')
# 				v[0] = v[0].strip().replace('\'', '')
# 				v[1] = v[1].strip().replace('\'', '')
# 				# dic[v[0]] = v[1]
# 				tmp_dict = {v[0]: v[1]}
# 				dict.update(tmp_dict)
# print('dict is ===============')
# print(dic)
import cv2
import numpy as np
import os

with open('savepara.txt', 'r') as f:
	datalist = f.readlines()
	f.close()
for no, index in enumerate(datalist):
	print(no, 'index', index)
	if '011' in index:
		del datalist[no: no+1]
with open('savepara.txt', 'w') as f:
	f.writelines(datalist)
	print('==========')

		# if txt2dic_tmp['image'] == com_imagename:
		# 	txt2dic.update(txt2dic_tmp)
