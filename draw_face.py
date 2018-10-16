"""
Average Face.

This script takes as command line argument the folder containing the
annotations of helen dataset and then creates a file "avg_shape.npy"
containing the average face of the dataset
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import List

if len(sys.argv) != 2:
    print("Run as following....")
    print("python3 calculate_average_face.py <path to anotations directory>")
    sys.exit(0)

if not os.path.exists(sys.argv[1]) or not os.path.isdir(sys.argv[1]):
    print("Directory doesn't exist or Invalid")
    sys.exit(0)

directory: str = sys.argv[1]
txtfiles: List[str] = os.listdir(directory)

df_list: List[np.ndarray] = []

for i in txtfiles:
    filename: str = os.path.join(directory, i)
    df: pd.core.frame.DataFrame = pd.read_csv(filename,
                                              sep=" , ",
                                              skiprows=1,
                                              header=None,
                                              engine='python')
    df_list.append(df.values)

print(len(df_list))
avg_shape: np.ndarray = np.zeros((194, 2))

for j in df_list:
    avg_shape += j

avg_shape /= len(df_list)
np.save("avg_shape.npy", avg_shape)
