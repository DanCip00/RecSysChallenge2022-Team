
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
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
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

urm = mm.defaultExplicitURM(urmv=URMv,urmo=URMo, normalize=False, add_aug=False)

urm.data = np.ones(len(urm.data))

URM_train_validation, URMv_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)
URMv_train, URMv_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.80)

urma = mm.augmentURM(URMv_train)
urma.data = np.ones(len(urma.data))
URMfat = sps.vstack([URMv_train,urma])           



################################### USER GROUP ######################################
profile_length = np.ediff1d(sps.csr_matrix(urm).indptr)
sorted_users = np.argsort(profile_length)
users_in_group = sorted_users[8*2081:]
users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]



evaluator_validation = EvaluatorHoldout(URMv_validation, [10]) #ignore_users=users_not_in_group
evaluator_test = EvaluatorHoldout(URMv_test, [10]) #ignore_users=users_not_in_group

metric_to_optimize = "MAP_MIN_DEN" 

recommender_class = RP3betaRecommender


hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)


#{'topK': 96, 'alpha': 0.602322918891714, 'beta': 0.30286490950247463, 'normalize_similarity': True} -> MAP 0.0276749
#{'topK': 93, 'alpha': 0.5450683986261076, 'beta': 0.33581240628495046, 'normalize_similarity': True} -> MAP 0.0276290
#{'topK': 106, 'alpha': 0.6155817289031643, 'beta': 0.33427474623540737, 'normalize_similarity': True -> MAP 0.0276355
#{'topK': 69, 'alpha': 0.6854042891733674, 'beta': 0.2763471245555947, 'normalize_similarity': True} ->  MAP 0.0282084

hyperparameters_range_dictionary = {
                "topK": Integer(50, 170),
                "alpha": Real(low = 0.4, high = 0.9, prior = 'uniform'),
                "beta": Real(low = 1e-2, high = 0.5, prior = 'uniform'),
                "normalize_similarity": Categorical([True]),
            }



recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URMv_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )

recommender_input_args_last_test = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train_validation],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {},
                EARLYSTOPPING_KEYWORD_ARGS = {},
            )


output_folder_path = "Daniele/Recommenders/RP3Beta/result_experiments_groups/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
#n_cases = 200  # using 10 as an example
n_cases = 15000
n_random_starts = int(n_cases*0.3)  
cutoff_to_optimize = 10

hyperparameterSearch.search(recommender_input_args,
                       recommender_input_args_last_test = recommender_input_args_last_test,
                       hyperparameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = n_random_starts,
                       save_model = "no",
                       output_folder_path = output_folder_path, # Where to save the results
                       output_file_name_root = "high_search",  # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )



