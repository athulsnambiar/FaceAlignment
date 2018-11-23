import cv2
import numpy as np
from shape_predictor import ShapePredictor
import sys
import os
import pickle as pkl

def load_dataset(path):
    data = os.listdir(path)
    imageNpArr = []
    landmarkNpArr=[]
    for i in range(len(data)):
        if os.path.splitext(data[i])[1] == '.npy':
            filename=os.path.splitext(data[i])[0]
            imagePath = path+"/"+filename+".png"
            landmrkPointsPath = path+"/"+filename+".npy"
            landmrkPointsNp = np.load(landmrkPointsPath)
            imageNp = cv2.imread(imagePath, 0)
            imageNpArr.append(imageNp)
            landmarkNpArr.append(landmrkPointsNp)

    return (imageNpArr,landmarkNpArr)


 if len(sys.argv) != 3:
     print("python3 demo.py train_folder test_folder")
     sys.exit(0)

images, landmark  = load_dataset(sys.argv[1])


print(type(landmark[0]))

sp = ShapePredictor(10, 500, tree_depth= 4, num_pixels=400, oversampling=10)
sp.train(images, landmark)
f = open("shape_predictor", "wb")
pkl.dump(sp, f)
f.close()

f = open("shape_predictor", "rb")
sp = pkl.load(f)
f.close()
images, landmark  = load_dataset(sys.argv[2])
error_arr = []

for i in range(len(images)):
    output = sp.predict(images[i])
    temp = landmark[i] - output
    temp = temp * temp
    temp = np.sqrt(np.sum(temp, axis = 1))
    error_arr.append(np.sum(temp))

error = sum(error_arr)/len(error_arr)
print(error)


output = sp.predict(images[i])
for point in output:
    cv2.circle(images[i], (int(point[0]),int(point[1])),2,(0,0,255),-1)

cv2.imshow('image', images[i])
cv2.waitKey(0)
