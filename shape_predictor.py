"""
Implement Shape Predictor Class.

Remove unrelated functions to utils.py
and remove this line
"""

import numpy as np
from typing import List
from forest import FernForest
from utilities import normalize_shape, denormalize_shape
# import sys


# this function is not complete. normalize mean after getting images
def compute_mean_shape(shapes: List[np.ndarray], size) -> np.ndarray:
    """Compute mean shape."""
    mean_shape: np.ndarray = np.zeros(shapes[0].shape)

    for i in shapes:
        mean_shape = mean_shape + i

    mean_shape = mean_shape / len(shapes)
    mean_shape = normalize_shape(mean_shape, size)
    return mean_shape


class ShapePredictor:
    """
    Shape Predictor.

    Overall class to handle all operations
    """

    def __init__(self,
                 num_stages: int = 10,
                 num_trees: int = 500,
                 tree_depth: int = 5,
                 num_pixels: int = 400,
                 oversampling: int = 10
                 ):
        """
        Initialize Class.

        num_stages: number of stages
        num_trees: number of ferns per stage
        num_landmarks: number of landmarks
        num_pixles: number of pixels to be selected as features
        tree_depth: depth of a single tree
        """
        self.num_stages: int = num_stages
        self.num_trees: int = num_trees
        self.tree_depth: int = tree_depth
        self.num_pixels: int = num_pixels
        self.oversampling: int = oversampling
        self.stages: List[FernForest] = []

    def train(self,
              images: List[np.ndarray],
              true_shapes: List[np.ndarray]):
        """
        Training Images.

        Steps:
        1) get mean image
        """
        if not images:
            raise ValueError("Empty Image array")

        if len(images) != len(true_shapes):
            raise ValueError("Image Array Length Not"
                             "Matching Shape Array Length")

        self.mean_shape: np.ndarray = compute_mean_shape(true_shapes, images[0].shape[0])
        self.training_shapes: List[np.ndarray] = true_shapes.copy()
        size = images[0].shape[0]

        sample_images: List[np.ndarray] = []
        sample_true_shape: List[np.ndarray] = []
        sample_error_shape: List[np.ndarray] = []

        for i in range(len(images)):
            for j in range(self.oversampling):
                # generate a index not equal to i
                index: int = np.random.randint(len(images))
                while index == i:
                    index = np.random.randint(len(images))

                sample_images.append(images[i])
                sample_true_shape.append(true_shapes[i])
                sample_error_shape.append(true_shapes[index])
        
        for i in range(self.num_stages):
            print("Training Stage ", i+1, "/", self.num_stages)
            # fern forest is not complete. add parameters
            self.stages.append(FernForest(i,
                                          self.num_trees,
                                          self.num_pixels,
                                          self.tree_depth))

            corrections: List[np.ndarray] = self.stages[i].\
                                            train(sample_images,
                                                  sample_true_shape,
                                                  sample_error_shape,
                                                  self.mean_shape)
            #not complete. normalize corrections and sample errors
            for j in range(len(corrections)):
                temp = normalize_shape(sample_error_shape[j], size) + corrections[j]
                sample_error_shape[j] = denormalize_shape(temp, size)


    def predict(self, image: np.ndarray):
        
        avg_prediction: np.ndarray = np.zeros(self.training_shapes[0].shape)
        size = image.shape[0]

        for i in range(self.oversampling):
            index = np.random.randint(len(self.training_shapes))
            # initial estimate
            predicted_shape = self.training_shapes[index]
            # not complete. shape normalization required
            for j in range(self.num_stages):
                correction = self.stages[j].predict(image,
                                                    self.mean_shape,
                                                    predicted_shape)
                predicted_shape = normalize_shape(predicted_shape, size) + correction
                predicted_shape = denormalize_shape(predicted_shape, size)
            
            avg_prediction = avg_prediction + predicted_shape
        
        return avg_prediction / self.oversampling
