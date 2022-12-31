import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

################################# IMPORT RECOMMENDERS #################################

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Daniele.Recommenders.KNN_CFCBF.ItemKNN_CFCBF_Hybrid_Recommender import KNN_CFCBF_custom
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.NonPersonalizedRecommender import TopPop

################################## IMPORT LIBRARIES ##################################

import pandas as pd
from numpy import linalg as LA
import Daniele.Utils.MatrixManipulation as mm 
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np


#################################### HYBRID CLASS ####################################

class SSLIMRP3BetaKNNRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "RP3BetaKNNRecommender"
    peso_1 = 0.6
    peso_2 = 0.4
    interactions_threshold = 10


    def __init__(self, urm_train):
        self.URM_train = urm_train

        super().__init__(self.URM_train)

    def fit(self, peso_1=0.6, peso_2=0.4):
        self.peso_1 = peso_1
        self.peso_2 = peso_2

        self.rp3beta_recommender = RP3betaRecommender(self.URM_train)
        self.sslim_recommender = SLIMElasticNetRecommender(URM_train=self.URM_train)
        self.most_viewed = TopPop(self.URM_train)

        self.most_viewed.fit()
        # {'topK': 69, 'alpha': 0.6854042891733674, 'beta': 0.2763471245555947, 'normalize_similarity': True} -> MAP 0.0282084
        self.rp3beta_recommender.fit( topK=69, alpha=0.6854042891733674, beta=0.2763471245555947, normalize_similarity=True )

        self.sslim_recommender.fit(alpha=0.003271, l1_ratio=0.006095, topK=884)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        w1 = self.rp3beta_recommender._compute_item_score(user_id_array)
        w1 = w1 / np.linalg.norm(w1, 2)

        w2 = self.sslim_recommender._compute_item_score(user_id_array)
        w2 = w2 / np.linalg.norm(w2, 2)

        item_weights = 0.6 * w2 + 0.4 * w1

        return item_weights