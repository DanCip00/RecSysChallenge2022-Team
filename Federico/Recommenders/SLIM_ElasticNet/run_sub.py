import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

################################# IMPORT RECOMMENDERS #################################
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

################################## IMPORT LIBRARIES ##################################
import scipy.sparse as sps
from tqdm import tqdm
import pandas as pd
import numpy as np
import similaripy
import math 

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm

URMv = dm.getURMviews()
URMo = dm.getURMopen()
URM_all = URMv + URMo
URM_all.data = np.ones(len(URM_all.data))

recommender = SLIMElasticNetRecommender(URM_all, verbose=True)
recommender.fit(alpha=0.003271, l1_ratio=0.006095, topK=884)


f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(dm.getUserIDtest_df().user_id):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")