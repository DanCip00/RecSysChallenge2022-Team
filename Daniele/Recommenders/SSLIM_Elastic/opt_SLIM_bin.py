import os
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm
import Daniele.Utils.SaveSparceMatrix as ssm

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize
import similaripy
import numpy as np

URMv = dm.getURMviews()
URMo = dm.getURMopen()
#URM_all = mm.defaultExplicitURM(urmv=URMv, urmo=URMo, normalize=False, add_aug=False)


ICMt=dm.getICMt()
ICMl=dm.getICMl()

path_save= "Daniele/Recommenders/SSLIM_Elastic/saved_models"
if not os.path.exists(path_save):
    os.makedirs(path_save)

name="train.csv"
dir = os.path.join(path_save,name)
if not os.path.exists(dir):

    URM_all = URMv + URMo
    URM_all.data = np.ones(len(URM_all.data))

    URM_train_val, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_val, train_percentage = 0.80)

    ssm.saveMatrix(dir,URM_train)

    name="URM_train_val.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URM_train_val)

    name="URM_validation.csv"
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

    name="URM_validation.csv"
    dir = os.path.join(path_save,name)
    URM_validation=ssm.readMatrix(dir)

    name="URM_train_val.csv"
    dir = os.path.join(path_save,name)
    URM_train_val=ssm.readMatrix(dir)



evaluator_validation = EvaluatorHoldout(URM_validation, [10])
evaluator_test = EvaluatorHoldout(URM_test, [10])

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
    CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_val],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={},
    EARLYSTOPPING_KEYWORD_ARGS={},
)

hyperparameters_range_dictionary = {
    "alpha": Real(low=1e-3, high=9e-3, prior='uniform'),
    "l1_ratio": Real(low=1e-3, high=9e-3, prior='uniform'),
    "topK": Integer(559, 1100),
}
output_folder_path = "Daniele/Recommenders/SSLIM_Elastic/result_experiments/"
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
    output_file_name_root="best_opt",  # How to call the files
    metric_to_optimize=metric_to_optimize,
    cutoff_to_optimize=cutoff_to_optimize,
    resume_from_saved=True
)