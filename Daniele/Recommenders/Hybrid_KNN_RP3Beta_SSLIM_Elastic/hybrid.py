import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

################################# IMPORT RECOMMENDERS #################################

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Daniele.Recommenders.KNN_CFCBF.ItemKNN_CFCBF_Hybrid_Recommender import KNN_CFCBF_custom
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
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


    def __init__(self, urm_train_bin,urm_train_exp):
        self.URM_train = urm_train_bin
        self.urm_train_exp = urm_train_exp

        super().__init__(self.URM_train)

    def fit(self, peso_1=0.6, peso_2=0.4,peso_3=0.6, peso_4=0.4,alpha=0.5):
        self.peso_1 = peso_1
        self.peso_2 = peso_2
        self.peso_3 = peso_3
        self.peso_4 = peso_4

        self.rp3beta_recommender = RP3betaRecommender(self.URM_train)
        self.slim_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=self.URM_train)
        self.sslim_BPR_recommender = SLIM_BPR_Cython(URM_train=self.urm_train_exp)
        self.most_viewed = TopPop(self.URM_train)
        self.slim_combo = ItemKNNCustomSimilarityRecommender(self.URM_train)

        path_save= "Daniele/Recommenders/Hybrid_KNN_RP3Beta_SSLIM_Elastic/saved_models"
        if not os.path.exists(path_save):
            os.makedirs(path_save)

    
        self.most_viewed.fit()
        # {'topK': 69, 'alpha': 0.6854042891733674, 'beta': 0.2763471245555947, 'normalize_similarity': True} -> MAP 0.0282084
        self.rp3beta_recommender.fit( topK=69, alpha=0.6854042891733674, beta=0.2763471245555947, normalize_similarity=True )
        
        #slim Elastic
        name="slim_Elastic"
        dir = os.path.join(path_save,name)

        self.slim_recommender = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=self.URM_train)
        if not os.path.exists(dir+".zip"):
            
            self.slim_recommender.fit(alpha=0.003271, l1_ratio=0.006095, topK=884)
            self.slim_recommender.save_model(path_save,name)
        else:
            self.slim_recommender.load_model(path_save,name)

        ### sslim bpr
        name="sslim01"
        dir = os.path.join(path_save,name)

        self.sslim_BPR_recommender = SLIM_BPR_Cython(URM_train=self.urm_train_exp)
        if not os.path.exists(dir+".zip"):
            #{'topK': 51, 'epochs': 15, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 1e-05, 'lambda_j': 0.003215687724797301, 'learning_rate': 0.007114410195895492} -> MAP 0.0078853
            self.sslim_BPR_recommender.fit(topK= 51, epochs=15,symmetric=True, sgd_mode = 'adam', lambda_i = 1e-05, lambda_j=0.003215687724797301, learning_rate = 0.007114410195895492)
            self.sslim_BPR_recommender.save_model(path_save,name)
        else:
            self.sslim_BPR_recommender.load_model(path_save,name)

        self.slim_combo.fit((1 - alpha) * self.slim_recommender.W_sparse + alpha* self.sslim_BPR_recommender.W_sparse)

        

    def _compute_item_score(self, user_id_array, items_to_compute=None):


        item_weights = np.empty([len(user_id_array), self.URM_train.shape[1]])
        for i in range(len(user_id_array)):

            interactions = len(self.URM_train[user_id_array[i],:].indices)
            if interactions >= 12 : #12 -> g. 7 
            
                w1 = self.rp3beta_recommender._compute_item_score([user_id_array[i]])
                w1 = w1 / np.linalg.norm(w1, 2)

                w2 = self.slim_recommender._compute_item_score([user_id_array[i]])
                w2 = w2 / np.linalg.norm(w2, 2)

                item_weights[i,:] = self.peso_1 * w2 + self.peso_2 * w1
                

            elif interactions >= 1:
                
                #w1 = self.sslim_BPR_recommender._compute_item_score([user_id_array[i]])
                w1 = self.rp3beta_recommender._compute_item_score([user_id_array[i]])
                w1 = w1 / np.linalg.norm(w1, 2)

                w2 = self.slim_combo._compute_item_score([user_id_array[i]])
                w2 = w2 / np.linalg.norm(w2, 2)

                item_weights[i,:] = self.peso_3 * w2 + self.peso_4 * w1

            else:
                print("topPop")
                w1 = self.most_viewed._compute_item_score([user_id_array[i]])
                item_weights[i,:] = w1 

        return item_weights