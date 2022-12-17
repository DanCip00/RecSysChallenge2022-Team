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
from Daniele.Recommenders.Hybrid_KNN_RP3Beta_SSLIM.hybrid import SSLIMRP3BetaKNNRecommender
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
import similaripy

URMv = dm.getURMviews()
URMo = dm.getURMopen()

ICMt=dm.getICMt()
ICMl=dm.getICMl()


URMv_train_validation, URMv_test = split_train_in_two_percentage_global_sample(URMv, train_percentage = 0.80)
URMv_train, URMv_validation = split_train_in_two_percentage_global_sample(URMv_train_validation, train_percentage = 0.80)



evaluator_validation = EvaluatorHoldout(URMv_validation, [10])
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
    CONSTRUCTOR_POSITIONAL_ARGS = [URMv_train,URMo,ICMt,ICMl],     # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS = {},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS = [URMv_train_validation,URMo,ICMt,ICMl],     
    CONSTRUCTOR_KEYWORD_ARGS = {},
    FIT_POSITIONAL_ARGS = [],
    FIT_KEYWORD_ARGS = {},
    EARLYSTOPPING_KEYWORD_ARGS= {},
)

"""
def fit(self, alpha_knn_rp3=0.3,
            
            topK_rp3beta= 89, alpha_rp3beta =0.6361002951626124, beta_rp3beta= 0.27432996564004203, normalize_similarity_rp3beta= True,
            topK_knn= 744, shrink_knn= 457, similarity_knn= 'cosine', normalize_knn= True, feature_weighting_knn='TF-IDF',
            topK_sslim= 305, epochs_sslim=25,symmetric_sslim=True, sgd_mode_sslim = 'adam', lambda_i_sslim = 0.0008132913041259862, lambda_j_sslim=0.004889521240194745, learning_rate_sslim = 0.005381553515814384,
            peso_1=0.6,peso_2=0.4,interactions_threshold =10

            ):
"""

"""
"topK_sslim": Integer(5, 750),
"epochs_sslim": Categorical([100]),
"symmetric_sslim": Categorical([True]),
"sgd_mode_sslim": Categorical(["adam", "adagrad"]), #"sgd", "adagrad"
"lambda_i_sslim": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
"lambda_j_sslim": Real(low = 1e-5, high = 1e-2, prior = 'log-uniform'),
"learning_rate_sslim": Real(low = 1e-4, high = 1e-1, prior = 'log-uniform'),
"""
hyperparameters_range_dictionary = {

    #"alpha_knn_rp3": Real(low = 0, high = 1, prior = 'uniform'),
    "topK_rp3beta": Integer(5, 150),
    "alpha_rp3beta": Real(low = 0, high = 1.5, prior = 'uniform'),
    "beta_rp3beta": Real(low = 0, high = 1.5, prior = 'uniform'),
    "normalize_similarity_rp3beta": Categorical([True]),

    "topK_knn": Integer(10, 750),
    "shrink_knn": Integer(0, 1000),
    "similarity_knn": Categorical(['cosine', 'jaccard', "asymmetric"]), #"dice", "tversky"
    "normalize_knn": Categorical([True]),
    "feature_weighting_knn": Categorical(["BM25", "TF-IDF"]),

    
    "peso_1": Real(low = 0, high = 1, prior = 'uniform'),
    "peso_2": Real(low = 0, high = 1, prior = 'uniform'),
    "interactions_threshold": Integer(2, 20),

}

output_folder_path = "Daniele/Recommenders/Hybrid_KNN_RP3Beta_SSLIM/result_experiments/"

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
                       output_file_name_root = "massive_optimization", # How to call the files
                       metric_to_optimize = metric_to_optimize,
                       cutoff_to_optimize = cutoff_to_optimize,
                      )


