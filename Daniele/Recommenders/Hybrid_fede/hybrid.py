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
    alpha = 0.2
    interactions_threshold = 15
    models_folder = "Daniele/Recommenders/Hybrid_fede/Models/"


    def __init__(self, urm_train):
        self.URM_train = urm_train

        super().__init__(self.URM_train)

    def fit(self, alpha=0.6, is_sub=False):
        if is_sub:
            self.models_folder = self.models_folder + 'sub/'

        self.alpha = alpha
        self.rp3beta_recommender = RP3betaRecommender(self.URM_train)
        self.sslim_recommender = SLIMElasticNetRecommender(URM_train=self.URM_train)
        self.most_viewed = TopPop(self.URM_train)

        if os.path.exists(self.models_folder + self.most_viewed.RECOMMENDER_NAME + ".zip"):
            self.most_viewed.load_model(self.models_folder)
        else:
            self.most_viewed.fit()
            self.most_viewed.save_model(self.models_folder)

        if os.path.exists(self.models_folder + self.rp3beta_recommender.RECOMMENDER_NAME + ".zip"):
            self.rp3beta_recommender.load_model(self.models_folder)
        else:
            self.rp3beta_recommender.fit(
                # {'topK': 69, 'alpha': 0.6854042891733674, 'beta': 0.2763471245555947, 'normalize_similarity': True} -> MAP 0.0282084
                topK=69, alpha=0.6854042891733674, beta=0.2763471245555947, normalize_similarity=True 
            )
            self.rp3beta_recommender.save_model(self.models_folder)

        if os.path.exists(self.models_folder + self.sslim_recommender.RECOMMENDER_NAME + ".zip"):
            self.sslim_recommender.load_model(self.models_folder)
        else:
            # {'alpha': 0.002930092866966509, 'l1_ratio': 0.006239337272696024, 'topK': 882} -> MAP 0.0422894 SLIM Elastic High
            self.sslim_recommender.fit(alpha=0.002930092866966509, l1_ratio=0.006239337272696024, topK=882)
            self.sslim_recommender.save_model(self.models_folder)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        w1 = self.rp3beta_recommender._compute_item_score(user_id_array)
        w1 = w1 / np.linalg.norm(w1, 2)

        w2 = self.sslim_recommender._compute_item_score(user_id_array)
        w2 = w2 / np.linalg.norm(w2, 2)

        weights = self._get_alpha_per_users(user_id_array)
        item_weights = w1
        for i in range(len(user_id_array)):
            item_weights[i] = weights[i] * w1[i] + (1 - weights[i]) * w2[i]

        return item_weights

    def _get_alpha_per_users(self, user_id_array):
        weights = np.zeros(len(user_id_array))

        for i in range(len(user_id_array)):
            num_interactions = len(self.URM_train[user_id_array[i],:].indices)
            if num_interactions < self.interactions_threshold:
                weights[i] = 1
            else:
                weights[i] = self.alpha

        return weights