"""Single Fern."""

import numpy as np
import math
import cv2
from typing import List


class Fern:
    """Fern."""

    def __init__(self, tree_depth):
        self.tree_depth: int = tree_depth
        self.threshold: List[int] = [] 
    
    def train(self,
              intensities: np.ndarray,
              pixel_covar: np.ndarray,
              test_pixels: List[np.ndarray],
              nearest_landmark: List[int],
              errors: List[np.ndarray]):
        
        self.num_landmark: int = errors[0].shape[0]
        num_pixels = len(test_pixels)

        self.selected_pixel_index: np.ndarray = np.zeros((self.tree_depth, 2))
        self.selected_pixel_loc: np.ndarray = np.zeros((self.tree_depth, 4))
        self.selected_nearest_landmark: np.ndarray = np.zeros((self.tree_depth, 2))

        for i in range(self.tree_depth):
            random_direction: np.ndarray = np.random.rand(self.num_landmark, 2)*2.2 - 1.1
            random_direction = cv2.normalize(random_direction, None)

            projection: np.ndarray = np.zeros((len(errors)))
            for j in range(len(errors)):
                projection[j] = np.sum(errors[j] * random_direction)

            covar_proj: np.ndarray = np.zeros((num_pixels))

            for j in range(num_pixels):
                # check here
                covar_proj[j] = np.cov(projection, intensities[j])[0, 1]
            
            max_correlation: float = -1
            max_pixel_1: int = 0
            max_pixel_2: int = 0
            
            for j in range(num_pixels):
                for k in range(num_pixels):
                    temp: float = (pixel_covar[j, j] + pixel_covar[k, k] -
                                   2*pixel_covar[j, k])

                    if abs(temp) < 1e-10:
                        continue
                    flag: bool = False

                    for p in range(i):
                        if (j == self.selected_pixel_index[p, 0] and 
                            k == self.selected_pixel_index[p, 1]):
                            flag = True
                            break
                        elif (j == self.selected_pixel_index[p, 1] and 
                            k == self.selected_pixel_index[p, 0]):
                            flag = True
                            break
                    
                    if flag:
                        continue
                    
                    temp1: float = (covar_proj[j] - covar_proj[k]) / math.sqrt(temp)
                    
                    if abs(temp1) > max_correlation:
                        max_correlation = temp1
                        max_pixel_1 = j
                        max_pixel_2 = k

            self.selected_pixel_index[i, 0] = max_pixel_1
            self.selected_pixel_index[i, 1] = max_pixel_2
            self.selected_pixel_loc[i, 0] = test_pixels[max_pixel_1][0, 0]
            self.selected_pixel_loc[i, 1] = test_pixels[max_pixel_1][0, 1]
            self.selected_pixel_loc[i, 2] = test_pixels[max_pixel_2][0, 0]
            self.selected_pixel_loc[i, 3] = test_pixels[max_pixel_2][0, 1]
            self.selected_nearest_landmark[i, 0] = nearest_landmark[max_pixel_1]
            self.selected_nearest_landmark[i, 1] = nearest_landmark[max_pixel_2]

            max_diff: float = -1
            for j in range(intensities.shape[1]):
                temp: float = (intensities[max_pixel_1, j] -
                                intensities[max_pixel_2, j])
                if abs(temp) > max_diff:
                    max_diff = abs(temp)
            
            self.threshold.append(np.random.uniform(-0.2*max_diff,
                                             0.2*max_diff))
        
        shapes_in_bins: List[List[int]] = []
        num_bins: int = 2**self.tree_depth
        for i in range(num_bins):
            shapes_in_bins.append([])

        for i in range(len(errors)):
            index: int = 0
            for j in range(self.tree_depth):
                intensity1: float = intensities[int(self.selected_pixel_index[j,0]), i]
                intensity2: float = intensities[int(self.selected_pixel_index[j,1]), i]
                if intensity1 - intensity2 >= self.threshold[j]:
                    index = index + 2**j
            
            shapes_in_bins[index].append(i)
        
        prediction: List[np.ndarray] = []
        for i in range(len(errors)):
            prediction.append(np.zeros(errors[0].shape))
        
        self.bin_output: List[np.ndarray] = []
        for i in range(num_bins):
            self.bin_output.append(np.zeros(errors[0].shape))
        
        for i in range(num_bins):
            temp: np.ndarray = np.zeros(errors[0].shape)
            bin_size: int = len(shapes_in_bins[i])

            for j in range(bin_size):
                index: int = shapes_in_bins[i][j]
                temp = temp + errors[index]
            
            if bin_size == 0:
                self.bin_output[i] = temp
                continue
            
            temp = (1/((1+ 1000/bin_size)*bin_size)) * temp
            self.bin_output[i] = temp

            for j in range(bin_size):
                index: int = shapes_in_bins[i][j]
                prediction[index] = temp
        
        return prediction

    def predict(self,
                image: np.ndarray,
                shape: np.ndarray,
                rotation: np.ndarray,
                scale: float):
        index: int = 0
        size: int = image.shape[0]

        for i in range(self.tree_depth):
            nearest_landmark_index_1: int = self.selected_nearest_landmark[i,0]
            nearest_landmark_index_2: int = self.selected_nearest_landmark[i,1]
            x: float = self.selected_pixel_loc[i,0]
            y: float = self.selected_pixel_loc[i,1]

            project_x: float = scale * (rotation[0, 0] * x + rotation[0, 1] * y) * size / 2 + shape[int(nearest_landmark_index_1), 0]
            project_y: float = scale * (rotation[1, 0] * x + rotation[1, 1] * y) * size / 2 + shape[int(nearest_landmark_index_1), 1]

            project_x = max(0, min(project_x, size-1))
            project_y = max(0, min(project_y, size-1))

            intensity1: float = int(image[int(project_y), int(project_y)])

            x = self.selected_pixel_loc[i,2]
            y = self.selected_pixel_loc[i,3]

            project_x = scale * (rotation[0, 0] * x + rotation[0, 1] * y) * size / 2 + shape[int(nearest_landmark_index_2), 0]
            project_y = scale * (rotation[1, 0] * x + rotation[1, 1] * y) * size / 2 + shape[int(nearest_landmark_index_2), 1]

            project_x = max(0, min(project_x, size-1))
            project_y = max(0, min(project_y, size-1))

            intensity2: float = int(image[int(project_y), int(project_y)])

            if intensity1 - intensity2 >= self.threshold[i]:
                index = index + 2**i
        
        return self.bin_output[index]
