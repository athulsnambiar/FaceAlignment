#ref https://www.geeksforgeeks.org/opencv-python-program-face-detection/

import sys
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from termcolor import colored


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
	for point in npa:
	        	cv2.circle(mask,(int(point[0]),int(point[1])),2,(0,0,255),-1)
	

	return mask,npa 



face_cascade_path = sys.argv[1]
image_folder_path = sys.argv[2]
output_folder_path = sys.argv[3]


face_cascade = cv2.CascadeClassifier(face_cascade_path)


image_files = [f for f in listdir(image_folder_path) if f.endswith('.png')]

for image_path in image_files:
	#read pts file for the image
	ptsfile = open(join(image_folder_path,image_path[:-4]+".pts"),'r')
	ptslist = [tuple(map(float,pt.strip().split())) for pt in ptsfile.readlines()[3:71] ]
	ptsfile.close()
	npa = np.array(ptslist)
	
	#read images here
	image = cv2.imread(join(image_folder_path, image_path))
#	for point in ptslist:
#	        cv2.circle(image,(int(point[0]),int(point[1])),1,(0,0,255))
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#detect faces here
	faces = face_cascade.detectMultiScale(gray_image,1.1,5)

	#display detection results
	for (x,y,w,h) in faces:
		#cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
		#roi_gray = gray_image[y:y+h,x:x+w]
		y1 = max(0,y-20)
		y2 = min(len(image),y+h+20)
		x1 = max(0,x-20)
		x2 = min(len(image[0]),x+w+20)
		npa[:,0] = npa[:,0]-x1;
		npa[:,1] = npa[:,1]-y1;
		roi_color = image[y1:y2,x1:x2]
		#display image
		processed_image,npa = get_square(roi_color,npa,500)
		cv2.imshow('image',processed_image)
		k=cv2.waitKey() & 0xff

		if k==27:
			cv2.imwrite(join(output_folder_path,str(image_path)+".png"), roi_color)
			np.save(join(output_folder_path,str(image_path)+".npy"),npa)
			print(colored('File '+image_path+' accepted','green'))
		else:
			print(colored('File '+image_path+' rejected','red'))

cv2.destroyAllWindows()




