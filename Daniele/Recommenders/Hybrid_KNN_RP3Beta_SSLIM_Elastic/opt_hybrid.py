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
from Daniele.Recommenders.Hybrid_KNN_RP3Beta_SSLIM_Elastic.hybrid import SSLIMRP3BetaKNNRecommender
from Recommenders.IR_feature_weighting import okapi_BM_25

################################## IMPORT LIBRARIES ##################################
import scipy.sparse as sps
from tqdm import tqdm
import pandas as pd
import numpy as np
import similaripy
import math 

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm
import Daniele.Utils.SaveSparceMatrix as ssm

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize
import similaripy

URMv = dm.getURMviews()
URMo = dm.getURMopen()
ICMt=dm.getICMt()
ICMl=dm.getICMl()

path_save= "Daniele/Recommenders/Hybrid_KNN_RP3Beta_SSLIM_Elastic/saved_models"
if not os.path.exists(path_save):
    os.makedirs(path_save)


name="train.csv"
dir = os.path.join(path_save,name)
if not os.path.exists(dir):
    URMv_train, URMv_test = split_train_in_two_percentage_global_sample(URMv, train_percentage = 0.80)

    ssm.saveMatrix(dir,URMv_train)

    name="test.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URMv_test)

    urm_def = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=True)
    name="urm_def.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,urm_def)

    urm_bin = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo, normalize=False, add_aug=True)
    urm_bin.data = np.ones(len(urm_bin.data))
    name="urm_bin.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,urm_bin)

else:
    URMv_train=ssm.readMatrix(dir)

    name="test.csv"
    dir = os.path.join(path_save,name)
    URMv_test=ssm.readMatrix(dir)

    name="urm_def.csv"
    dir = os.path.join(path_save,name)
    urm_def = ssm.readMatrix(dir)

    name="urm_bin.csv"
    dir = os.path.join(path_save,name)
    urm_bin = ssm.readMatrix(dir)





evaluator_validation = EvaluatorHoldout(URMv_test, [10])
evaluator_test = EvaluatorHoldout(URMv_test, [10])

metric_to_optimize = "MAP_MIN_DEN" 

recommender_class = SSLIMRP3BetaKNNRecommender

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)


earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 4,
                          "validation_metric": metric_to_optimize,
                          }

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_bin,urm_def],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_bin,urm_def],     
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS= {},
)

hyperparameters_range_dictionary = {

    #"alpha_knn_rp3": Real(low = 0, high = 1, prior = 'uniform'),
    
    "peso_1": Real(low = 1e-1, high = 1, prior = 'uniform'),
    "peso_2": Real(low = 1e-1, high = 1, prior = 'uniform'),
    "peso_3": Real(low = 1e-1, high = 1, prior = 'uniform'),
    "peso_4": Real(low = 1e-1, high = 1, prior = 'uniform'),
    "alpha": Real(low = 1e-1, high = 1, prior = 'uniform'),
}

output_folder_path = "Daniele/Recommenders/Hybrid_KNN_RP3Beta_SSLIM_Elastic/result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
#n_cases = 200  # using 10 as an example
n_cases = 50000
n_random_starts = int(n_cases*0.3)  
cutoff_to_optimize = 10

hyperparameterSearch.search(recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "no",
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = "params", # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


