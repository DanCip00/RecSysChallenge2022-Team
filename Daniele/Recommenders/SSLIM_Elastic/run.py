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
import numpy as np
from tqdm import tqdm

URMv = dm.getURMviews()
URMo = dm.getURMopen()
URM_all = mm.defaultExplicitURM(urmv=URMv, urmo=URMo, normalize=True, add_aug=False)
ICMt=dm.getICMt()
ICMl=dm.getICMl()





from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender

recommender = MultiThreadSLIM_SLIMElasticNetRecommender



recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all)

#{'alpha': 0.00403108385692859, 'l1_ratio': 0.005489484881997955, 'topK': 842} -> MAP 0.0285183
recommender.fit(topK= 842, alpha= 0.00403108385692859, l1_ratio= 0.005489484881997955,workers=3)



f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(dm.getUserIDtest_df().user_id):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")