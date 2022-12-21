
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
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
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


from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize

URMv = dm.getURMviews()
URMo = dm.getURMopen()
ICMt = dm.getICMt()
ICMl = dm.getICMl()


URMv_train_validation, URMv_test = split_train_in_two_percentage_global_sample(URMv, train_percentage = 0.75)
URMv_train, URMv_validation = split_train_in_two_percentage_global_sample(URMv_train_validation, train_percentage = 0.75)

urm_def = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=True)
urm_def_test = mm.defaultExplicitURM(urmv=URMv_train_validation,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=True)
#urm_bin = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo, normalize=False, add_aug=True)


################################### USER GROUP ######################################
profile_length = np.ediff1d(sps.csr_matrix(URMv_train).indptr)
sorted_users = np.argsort(profile_length)
users_in_group = sorted_users[0:int(10*(URMv_train.shape[0]/20))]
users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]



evaluator_validation = EvaluatorHoldout(URMv_validation, [10], ignore_users=users_not_in_group)
evaluator_test = EvaluatorHoldout(URMv_test, [10], ignore_users=users_not_in_group)

metric_to_optimize = "MAP_MIN_DEN" 

recommender_class = SLIM_BPR_Cython

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
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_def],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {"positive_threshold_BPR": None,
                        'train_with_sparse_weights': False,
                        'allow_train_with_sparse_weights': False},
    EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [urm_def_test],
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {"positive_threshold_BPR": None,
                        'train_with_sparse_weights': False,
                        'allow_train_with_sparse_weights': False},
    EARLYSTOPPING_KEYWORD_ARGS = earlystopping_keywargs,
)

#{'topK': 44, 'epochs': 25, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.01, 'lambda_j': 1e-05, 'learning_rate': 0.01}
#{'topK': 45, 'epochs': 25, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.01, 'lambda_j': 4.617001976423661e-05, 'learning_rate': 0.01}

hyperparameters_range_dictionary = {
                "topK": Integer(5, 750),
                "epochs": Categorical([800]),
                "symmetric": Categorical([True]),
                "sgd_mode": Categorical(["adam"]), #"sgd", "adagrad"
                "lambda_i": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
                "lambda_j": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
                "learning_rate": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
            }


output_folder_path = "Daniele/Recommenders/SSLIM/result_experiments/"

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


