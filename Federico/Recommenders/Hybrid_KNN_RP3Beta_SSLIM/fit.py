
import os
import sys

while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

from Federico.Recommenders.Hybrid_KNN_RP3Beta_SSLIM.hybrid import SSLIMRP3BetaKNNRecommender

################################# IMPORT HyperTuning  #################################
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical

################################# IMPORT RECOMMENDERS #################################
#from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.IR_feature_weighting import okapi_BM_25

################################## IMPORT LIBRARIES ##################################
import scipy.sparse as sps
from tqdm import tqdm
import pandas as pd
import numpy as np
import similaripy
import math
import os

import Daniele.Utils.MyDataManager as dm
import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.SaveSparceMatrix as ssm


from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


path_save = "Federico/Recommenders/Hybrid_KNN_RP3Beta_SSLIM/matrix"
if not os.path.exists(path_save):
    os.makedirs(path_save)

name="train.csv"
dir = os.path.join(path_save,name)
if not os.path.exists(dir):
    URMv = dm.getURMviews()
    URMo = dm.getURMopen()
    URM_all = URMv + URMo
    URM_all.data = np.ones(len(URM_all.data))

    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.85)
    ssm.saveMatrix(dir,URM_train)

    name="validation.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URM_validation)

    name="test.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URM_test)

else:
    URM_train=ssm.readMatrix(dir)

    name="test.csv"
    dir = os.path.join(path_save,name)
    URM_test=ssm.readMatrix(dir)

    name="validation.csv"
    dir = os.path.join(path_save,name)
    URM_validation=ssm.readMatrix(dir)

recommender = SSLIMRP3BetaKNNRecommender(urm_train=URM_train)
recommender.fit()
