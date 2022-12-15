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
from Daniele.Recommenders.KNN_CFCBF.ItemKNN_CFCBF_Hybrid_Recommender import KNN_CFCBF_custom
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

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize

URMv = mm.explicitURM(dm.getURMviews(),slope=0.01, n_remove=2750, shrink_bias=85,bias='item', new_val = 0)
URMo = mm.explicitURM(dm.getURMopen(), slope=0.01, n_remove = 10000, shrink_bias = 25,bias='user', new_val = 30)
icm = mm.augmentedICM(dm.getICMt(), dm.getICMl())



URMo_train_validation, URMo_test = split_train_in_two_percentage_global_sample(URMo, train_percentage = 0.80)
URMo_train, URMo_validation = split_train_in_two_percentage_global_sample(URMo_train_validation, train_percentage = 0.80)

    


################################### USER GROUP ######################################
"""
Recommender ottimizzato per users con poche views e tante "aperture"
"""
profile_length = np.ediff1d(sps.csr_matrix(URMo).indptr)
sorted_users = np.argsort(profile_length)
users_in_group = sorted_users[2081*10:]
users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]



evaluator_validation = EvaluatorHoldout(URMo_validation, [10],ignore_users=users_not_in_group)
evaluator_test = EvaluatorHoldout(URMo_test, [10], ignore_users=users_not_in_group)

metric_to_optimize = "MAP_MIN_DEN" 

recommender_class = KNN_CFCBF_custom

hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                         evaluator_validation=evaluator_validation,
                                         evaluator_test=evaluator_test)

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URMv,URMo_train,icm],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URMv,URMo_train_validation,icm],     
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

hyperparameters_range_dictionary = {
    "topK": Integer(10, 750),
    "shrink": Integer(0, 1000),
    "similarity": Categorical(['cosine', 'jaccard', "asymmetric"]), #"dice", "tversky"
    "normalize": Categorical([True]),
    "feature_weighting": Categorical(["BM25", "TF-IDF"]),

    "beta": Real(low = 0.5, high = 1.5, prior = 'uniform'),
    "ICM_weight": Real(low = 0.5, high = 1.5, prior = 'uniform'),
}

output_folder_path = "Daniele/Recommenders/KNN_CFCBF/result_experiments/"

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
                       output_file_name_root = "opens_recommender", # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


