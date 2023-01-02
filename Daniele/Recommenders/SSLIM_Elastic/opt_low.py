import os
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize
import similaripy
import scipy.sparse as sps
import numpy as np

URMv = dm.getURMviews()
URMo = dm.getURMopen()
#URM_all = mm.defaultExplicitURM(urmv=URMv, urmo=URMo, normalize=False, add_aug=False)

URM_all = URMv + URMo
URM_all.data = np.ones(len(URM_all.data))


ICMt=dm.getICMt()
ICMl=dm.getICMl()

URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.80)


icma = similaripy.normalization.bm25plus(mm.augmentedICM(ICMt, ICMl))
URM_train = sps.vstack([URM_train,icma.T])

################################### USER GROUP ######################################
profile_length = np.ediff1d(sps.csr_matrix(URM_all).indptr)
sorted_users = np.argsort(profile_length)
users_in_group = sorted_users[:8*2081]
users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]



evaluator_validation = EvaluatorHoldout(URM_validation, [10],ignore_users=users_not_in_group) #ignore_users=users_not_in_group
evaluator_test = EvaluatorHoldout(URM_test, [10],ignore_users=users_not_in_group) #ignore_users=users_not_in_group

from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender

recommender = MultiThreadSLIM_SLIMElasticNetRecommender
#recommender.fit(alpha=0.080068, l1_ratio=0.004213)

from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical

metric_to_optimize = "MAP_MIN_DEN"

hyperparameterSearch = SearchBayesianSkopt(recommender,
                                           evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)


recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],  # For a CBF model simply put [URM_train, ICM_train]
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

recommender_input_args_last_test = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

hyperparameters_range_dictionary = {
    "alpha": Real(low=1e-3, high=9e-3, prior='uniform'),
    "l1_ratio": Real(low=1e-3, high=9e-3, prior='uniform'),
    "topK": Integer(850, 1100),
}
output_folder_path = "Daniele/Recommenders/SSLIM_Elastic/result_experiments/daniele/"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 2000
n_random_starts = int(n_cases * 0.3)
cutoff_to_optimize = 10

hyperparameterSearch.search(
    recommender_input_args,
    recommender_input_args_last_test=recommender_input_args_last_test,
    hyperparameter_search_space=hyperparameters_range_dictionary,
    n_cases=n_cases,
    n_random_starts=n_random_starts,
    save_model="best",
    output_folder_path=output_folder_path,  # Where to save the results
    output_file_name_root="gruppi_basso",  # How to call the files
    metric_to_optimize=metric_to_optimize,
    cutoff_to_optimize=cutoff_to_optimize,
    resume_from_saved=True
)