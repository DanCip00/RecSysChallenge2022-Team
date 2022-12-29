import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

################################# IMPORT RECOMMENDERS #################################
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

################################## IMPORT LIBRARIES ##################################
import scipy.sparse as sps
from tqdm import tqdm
import pandas as pd
import numpy as np
import similaripy
import math 

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm


URM_all = mm.defaultExplicitURM(urmv=dm.getURMviews(), urmo=dm.getURMopen(), normalize=False, add_aug=True)

recommender = SLIM_BPR_Cython(URM_train=URM_all, verbose=False)
recommender.fit(epochs=10, symmetric=True, lambda_i=0.552113, lambda_j=0.444161, learning_rate=0.02174)


f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(dm.getUserIDtest_df().user_id):
    print("ciao: ", t)
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")