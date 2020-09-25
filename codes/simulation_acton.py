#!/usr/bin/env python3

# This script simulates batch bandits

# direct to parent folder
import sys
# sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import defaultdict
import math
import json

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, DotProduct, RBF 
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn_extra.cluster import KMedoids

from kernels_for_GPK import *


sys.path.insert(1, '../acton')

from acton.acton import predict, recommend, label
import acton.database
from acton.proto.wrappers import Recommendations



# # Initial labels.
# recommendation_indices = list(range(10))
# with acton.database.ASCIIReader(
#         'tests/data/classification.txt',
#         feature_cols=[], label_col='col20') as db:
#     recommendations = Recommendations.make(
#         recommended_ids=recommendation_indices,
#         labelled_ids=[],
#         recommender='None',
#         db=db)
# labels = label(recommendations)

# # Main loop.
# for epoch in range(10):
#     print('Epoch', epoch)
#     labels = label(
#         recommend(predict(labels, 'LogisticRegression'), 'RandomRecommender'))

# print('Labelled instances:', labels.ids)