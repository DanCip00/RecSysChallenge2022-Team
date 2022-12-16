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

import Daniele.Utils.MatrixManipulation as mm
import MyDataManager as dm

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize

URMv = dm.getURMviews()
URMo = dm.getURMopen()

urm = mm.defaultExplicitURM(urmv=URMv,urmo=URMo, normalize=False, add_aug=False)
urm.data = np.ones(len(urm.data))

#URMv_train, URMv_test = split_train_in_two_percentage_global_sample(urm, train_percentage = 0.80)
URMv_train = urm 
urma = mm.augmentURM(URMv_train)
urma.data = np.ones(len(urma.data))
URMfat = sps.vstack([URMv_train,urma])    


recommender = RP3betaRecommender(URMfat)
#{'topK': 106, 'alpha': 0.6155817289031643, 'beta': 0.33427474623540737, 'normalize_similarity': True} -> MAP 0.0276355
recommender.fit(topK= 106, alpha= 0.6155817289031643, beta= 0.33427474623540737, normalize_similarity= True)


f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(dm.getUserIDtest_df().user_id):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")