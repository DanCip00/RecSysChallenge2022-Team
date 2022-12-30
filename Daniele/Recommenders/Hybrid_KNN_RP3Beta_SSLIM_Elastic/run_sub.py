import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

################################# IMPORT RECOMMENDERS #################################
from Daniele.Recommenders.Hybrid_KNN_RP3Beta_SSLIM_Elastic.hybrid import SSLIMRP3BetaKNNRecommender

################################## IMPORT LIBRARIES ##################################
import scipy.sparse as sps
from tqdm import tqdm
import pandas as pd
import numpy as np
import math

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm


URMv = dm.getURMviews()
URMo = dm.getURMopen()
ICMt = dm.getICMt()
ICMl = dm.getICMl()
URM_all = URMv + URMo
URM_all.data = np.ones(len(URM_all.data))

recommender = SSLIMRP3BetaKNNRecommender(urm_train_bin=URM_all,urm_train_exp=mm.defaultExplicitURM(urmv=URMv,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=True))
recommender.fit()


f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(dm.getUserIDtest_df().user_id):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")