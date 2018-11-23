
import sys
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
# from termcolor import colored


def get_square(image,npa,square_size):
	
	height,width,dim=image.shape
	if(height>width):
		differ=height
	else:
		differ=width

	mask = np.zeros((differ,differ,dim), dtype="uint8")   
	x_pos=int((differ-width)/2)
	y_pos=int((differ-height)/2)
	npa[:,0] = npa[:,0]+x_pos;
	npa[:,1] = npa[:,1]+y_pos;
	
	npa[:,0] = npa[:,0]*(square_size/differ);
	npa[:,1] = npa[:,1]*(square_size/differ);

	mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]	
	mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)
	#for point in npa:
	#        	cv2.circle(mask,(int(point[0]),int(point[1])),2,(0,0,255),-1)
	

	return mask,npa 


image_folder_path = sys.argv[1]
output_folder_path = sys.argv[2]
oversize_factor = float(sys.argv[3])

image_files = [f for f in listdir(image_folder_path) if f.endswith('.png')]

pro_count=0
for image_path in image_files:
	#read pts file for the image
	ptsfile = open(join(image_folder_path,image_path[:-4]+".pts"),'r')
	ptslist = [tuple(map(float,pt.strip().split())) for pt in ptsfile.readlines()[3:71] ]
	ptsfile.close()
	npa = np.array(ptslist)
	center = npa.mean(axis=0)	
	diff = npa-center
	dist = (diff[:,0]**2 +diff[:,1]**2)**(1/2) 
	max_dist = max(dist)
	half_side = max_dist*(oversize_factor)
	image = cv2.imread(join(image_folder_path, image_path))

	x1 = int(max(0,center[0]-half_side))
	y1 = int(max(0,center[1]-half_side))
	x2 = int(min(center[0]+half_side,len(image[0])))
	y2 = int(min(center[1]+half_side,len(image)))
	#read images here

	npa[:,0] = npa[:,0]-x1;
	npa[:,1] = npa[:,1]-y1;
	roi_color = image[y1:y2,x1:x2]
	#get same res image
	processed_image,npa = get_square(roi_color,npa,500)
	pro_count += 1
	cv2.imwrite(join(output_folder_path,str(image_path)+".png"), processed_image)
	np.save(join(output_folder_path,str(image_path)+".npy"),npa)

	if pro_count%100==0:
		print('Processed '+str(pro_count)+' images')





