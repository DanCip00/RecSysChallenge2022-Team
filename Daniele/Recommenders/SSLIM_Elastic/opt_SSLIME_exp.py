
import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

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

URMv = dm.getURMviews()
URMo = dm.getURMopen()
ICMt = dm.getICMt()
ICMl = dm.getICMl()

path_save= "Daniele/Recommenders/SSLIM_Elastic/saved_models"
if not os.path.exists(path_save):
    os.makedirs(path_save)

name="train.csv"
dir = os.path.join(path_save,name)
if not os.path.exists(dir):
    URMv_train_val, URMv_test = split_train_in_two_percentage_global_sample(URMv, train_percentage = 0.80)
    URMv_train, URMv_validation = split_train_in_two_percentage_global_sample(URMv_train_val, train_percentage = 0.80)

    ssm.saveMatrix(dir,URMv_train)

    name="URMv_validation.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URMv_validation)

    name="test.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URMv_test)

    urm_def = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=True)
    name="urm_def.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,urm_def)

    urm_def_val = mm.defaultExplicitURM(urmv=URMv_train_val,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=True)
    name="urm_def_val.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,urm_def_val)


else:
    URMv_train=ssm.readMatrix(dir)

    name="test.csv"
    dir = os.path.join(path_save,name)
    URMv_test=ssm.readMatrix(dir)

    name="URMv_validation.csv"
    dir = os.path.join(path_save,name)
    URMv_validation=ssm.readMatrix(dir)

    name="urm_def.csv"
    dir = os.path.join(path_save,name)
    urm_def = ssm.readMatrix(dir)

    name="urm_def_val.csv"
    dir = os.path.join(path_save,name)
    urm_def_val = ssm.readMatrix(dir)



"""
################################### USER GROUP ######################################
profile_length = np.ediff1d(sps.csr_matrix(URMv_train).indptr)
sorted_users = np.argsort(profile_length)
users_in_group = sorted_users[0:int(10*(URMv_train.shape[0]/20))]
users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]
"""


evaluator_validation = EvaluatorHoldout(URMv_validation, [10]) # ignore_users=users_not_in_group
evaluator_test = EvaluatorHoldout(URMv_test, [10]) #ignore_users=users_not_in_group

metric_to_optimize = "MAP_MIN_DEN" 

recommender_class = MultiThreadSLIM_SLIMElasticNetRecommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)


recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_def],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_def_val],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

hyperparameters_range_dictionary = {
                "topK": Integer(20, 1500),
                "l1_ratio": Real(low = 1e-6, high = 1e-3, prior = 'log-uniform'),
                "alpha": Real(low = 1e-1, high = 10, prior = 'uniform'),
                "workers":Categorical([5]),
            }


output_folder_path = "Daniele/Recommenders/SSLIM_Elastic/result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
#n_cases = 200  # using 10 as an example
n_cases = 1000
n_random_starts = int(n_cases*0.3)  
cutoff_to_optimize = 10

hyperparameterSearch.search(recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "no",
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = "explicit_matrix", # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )