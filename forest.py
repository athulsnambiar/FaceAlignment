"""Fern Forest Class."""

import numpy as np
import cv2
import scipy.spatial
import math
from typing import List
from fern import Fern
from utilities import normalize_shape, denormalize_shape

# move this to utilities.py
# add normalization
def similarity_transform(shape1: np.ndarray,
                         shape2: np.ndarray):
    # 2d rotation matrix
    rotation: np.ndarray = np.zeros((2, 2))
    scale: float = 0

    mean1 = np.mean(shape1, axis=0)
    mean2 = np.mean(shape2, axis=0)

    mean_shifted_1: np.ndarray = shape1 - mean1
    mean_shifted_2: np.ndarray = shape2 - mean2

    cov1 = np.cov(mean_shifted_1, rowvar=False)
    cov2 = np.cov(mean_shifted_2, rowvar=False)

    s1: float = math.sqrt(np.linalg.norm(cov1))
    s2: float = math.sqrt(np.linalg.norm(cov2))

    scale = s1 / s2

    mean_shifted_1 = mean_shifted_1 / s1
    mean_shifted_2 = mean_shifted_2 / s2

    side1: float = 0
    side2: float = 0
    for i in range(shape1.shape[0]):
        side1 += (mean_shifted_1[i, 1] * mean_shifted_2[i, 0] -
                  mean_shifted_1[i, 0] * mean_shifted_2[i, 1])
        side2 += (mean_shifted_1[i, 0] * mean_shifted_2[i, 0] +
                  mean_shifted_1[i, 1] * mean_shifted_2[i, 1])

    hypot: float = math.sqrt(side1*side1 + side2*side2)
    sine: float = side1 / hypot
    cosine: float = side2 / hypot

    rotation[0, 0] = cosine
    rotation[0, 1] = -sine
    rotation[1, 0] = sine
    rotation[1, 1] = cosine

    return rotation, scale


class FernForest:
    """Forest of Ferns."""

    def __init__(self,
                 stage_index: int,
                 num_trees: int = 500,
                 num_pixels: int = 400,
                 tree_depth: int = 5):
        """Initialize forest."""

        self.num_trees: int = num_trees
        self.num_pixels: int = num_pixels
        self.tree_depth: int = tree_depth
        self.stage_index: int = stage_index
        self.ferns: List[Fern] = []

    def train(self,
              images: List[np.ndarray],
              true_shapes: List[np.ndarray],
              current_prediction: List[np.ndarray],
              mean_shape: np.ndarray):
        """One Stage of Regressor"""

        size = images[0].shape[0]

        error: List[np.ndarray] = []
        # not complete. normalization required.
        for i in range(len(current_prediction)):
            temp: np.ndarray = (normalize_shape(true_shapes[i], size) -
                                normalize_shape(current_prediction[i], size))

            rotation, scale = similarity_transform(mean_shape,
                                                   normalize_shape(current_prediction[i], size))
            # confusion here about the transpose of rotation matrix
            rotation = rotation.T
            error.append(scale * np.dot(temp, rotation))
        
        # generate random points
        # put in seperate function if possible
        test_pixels: List[np.ndarray] = []
        shape_index: List[int] = []
        for i in range(self.num_pixels):
            random_point: np.ndarray = np.random.rand(1, 2) * 2 - 1

            while np.linalg.norm(random_point) > 1:
                random_point: np.ndarray = np.random.rand(1, 2) * 2 - 1

            dist: np.ndarray = scipy.spatial.distance.cdist(mean_shape,
                                                            random_point)
            index: int = np.argmin(dist)
            random_point = random_point - mean_shape[index]
            test_pixels.append(random_point)
            shape_index.append(index)
        
        # compute pixel intensities at pixel locations computed above
        # not complete normalization required
        # move to seperate function if possible
        intensities: np.ndarray = np.empty((self.num_pixels, 0))
        for i in range(len(images)):
            norm = normalize_shape(current_prediction[i], size)
            rotation, scale = similarity_transform(norm, mean_shape)

            temp: np.ndarray = np.empty((0, 1))
            for j in range(self.num_pixels):
                point: np.ndarray = np.dot(test_pixels[j], rotation) * scale
                point = point * size / 2
                point = point + current_prediction[i][shape_index[j]]
                
                point[0, 0] = max(0, min(point[0, 0], size - 1))
                point[0, 1] = max(0, min(point[0, 1], size - 1))
                
                point = np.reshape(point, (2))
                temp = np.row_stack((temp,
                                        [
                                            [images[i][int(point[1]), int(point[0])]]
                                        ]))
            intensities = np.column_stack((intensities, temp))
        
        # compute covariance of intensities for better feature selection
        pixel_covar = np.cov(intensities, rowvar=True)

        prediction: List[np.ndarray] = []
        for i in range(len(images)):
            prediction.append(np.zeros(mean_shape.shape))

        for i in range(self.num_trees):
            print("Training Stage: ", self.stage_index,
                  " Fern: ", i)
            self.ferns.append(Fern(self.tree_depth))
            temp = self.ferns[i].train(intensities, pixel_covar, test_pixels,
                                       shape_index, error)
            for j in range(len(temp)):
                prediction[j] = prediction[j] + temp[j]
                error[j] = error[j] - temp[j]
        
        for i in range(len(prediction)):
            temp = normalize_shape(current_prediction[i], size)
            rotation, scale = similarity_transform(temp,
                                                   mean_shape)
            prediction[i] = scale * np.dot(prediction[i], rotation)
        
        return prediction

    #not complete. normalization required.
    def predict(self,
                image: np.ndarray,
                mean_shape: np.ndarray,
                current_prediction: np.ndarray):
        
        prediction: np.ndarray = np.zeros(current_prediction.shape)
        size = image.shape[0]
        temp = normalize_shape(current_prediction, size)
        rotation, scale = similarity_transform(temp,
                                               mean_shape)
        for i in range(self.num_trees):
            prediction = prediction + self.ferns[i].predict(image,
                                                            current_prediction,
                                                            rotation,
                                                            scale)
        temp = normalize_shape(current_prediction, size)
        rotation, scale = similarity_transform(temp,
                                               mean_shape)
        rotation = rotation.T
        prediction = scale * np.dot(prediction, rotation)
        return prediction
