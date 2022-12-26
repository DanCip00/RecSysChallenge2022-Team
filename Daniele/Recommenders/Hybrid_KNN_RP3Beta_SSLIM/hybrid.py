import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

################################# IMPORT RECOMMENDERS #################################

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Daniele.Recommenders.KNN_CFCBF.ItemKNN_CFCBF_Hybrid_Recommender import KNN_CFCBF_custom
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
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


    def __init__(self, urmv,urmo,icmt,icml):
        
        self.urmv = mm.explicitURM(urmv,slope=0.01, n_remove=2750, shrink_bias=85,bias='item', new_val = 0)
        self.urmo = mm.explicitURM(urmo, slope=0.01, n_remove = 10000, shrink_bias = 25,bias='user', new_val = 30)
        self.ICM_type = mm.augmentedICM(icmt,icml)

        urm = mm.defaultExplicitURM(urmv=urmv,urmo=urmo, normalize=False, add_aug=True)
        urm.data = np.ones(len(urm.data))
        self.urm_bin = urm

        self.urm_def = mm.defaultExplicitURM(urmv=urmv,urmo=urmo,icml=icml,icmt=icmt, normalize=True, add_aug=True,appendICM=True)

        self.interactions_threshold = 21
        super().__init__(self.urm_bin)

    def fit(self, alpha_knn_rp3=0.3,
            
            topK_rp3beta= 89, alpha_rp3beta =0.6361002951626124, beta_rp3beta= 0.27432996564004203, normalize_similarity_rp3beta= True,
            topK_knn= 744, shrink_knn= 457, similarity_knn= 'cosine', normalize_knn= True, feature_weighting_knn='TF-IDF',
            topK_sslim= 305, epochs_sslim=25,symmetric_sslim=True, sgd_mode_sslim = 'adam', lambda_i_sslim = 0.00048157278406027107, lambda_j_sslim=0.0002827394953195856, learning_rate_sslim = 0.009845463659115065,
            peso_1=0.6,peso_2=0.4,interactions_threshold =12

            ):
        self.peso_1 = peso_1
        self.peso_2 = peso_2
        self.interactions_threshold = interactions_threshold

        self.rp3beta_recommender = RP3betaRecommender(self.urm_bin)
        self.KNN_recommender = KNN_CFCBF_custom(self.urmv,self.urmo,ICM_train=self.ICM_type)
        self.sslim_recommender = SLIM_BPR_Cython(URM_train=self.urm_def)
        self.knn_rp3_recommender = ItemKNNCustomSimilarityRecommender(self.urm_def)
        self.most_viewed = TopPop(self.urm_bin)

        self.most_viewed.fit()
        # {'topK': 89, 'alpha': 0.6361002951626124, 'beta': 0.27432996564004203, 'normalize_similarity': True} -> opt_top
        self.rp3beta_recommender.fit(topK= topK_rp3beta, alpha= alpha_rp3beta, beta= beta_rp3beta, normalize_similarity= normalize_similarity_rp3beta)
        
        # {'topK': 744, 'shrink': 457, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'} -> opt_top
        self.KNN_recommender.fit(topK= topK_knn, shrink= shrink_knn, similarity= similarity_knn, normalize= normalize_knn, feature_weighting=feature_weighting_knn)

        # {'topK': 305, 'epochs': 25, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.0008132913041259862, 'lambda_j': 0.004889521240194745, 'learning_rate': 0.005381553515814384}
        # {'topK': 109, 'epochs': 25, 'symmetric': True, 'sgd_mode': 'adam', 'lambda_i': 0.00048157278406027107, 'lambda_j': 0.0002827394953195856, 'learning_rate': 0.009845463659115065}
        self.sslim_recommender.fit(topK= topK_sslim, epochs=epochs_sslim,symmetric=symmetric_sslim, sgd_mode = sgd_mode_sslim, lambda_i = lambda_i_sslim, lambda_j=lambda_j_sslim, learning_rate = learning_rate_sslim)

        # alpha =0.3
        self.knn_rp3_recommender.fit((1 - alpha_knn_rp3) * self.KNN_recommender.W_sparse + alpha_knn_rp3* self.rp3beta_recommender.W_sparse)

    def _compute_item_score(self,
                            user_id_array,
                            items_to_compute=None
                            ):


        item_weights = np.empty([len(user_id_array), self.urm_def.shape[1]])
        for i in range(len(user_id_array)):

            interactions = len(self.urmv[user_id_array[i],:].indices)

            if interactions >= 22:  # -> g.14
            
                w1 = self.rp3beta_recommender._compute_item_score([user_id_array[i]])
                #w1 = w1 / np.linalg.norm(w1, 2)
                item_weights[i,:] = w1 

            elif interactions >= 15: # -> g. 10

                w1 = self.knn_rp3_recommender._compute_item_score([user_id_array[i]])
                item_weights[i,:] = w1

            elif interactions >= self.interactions_threshold: #12 -> g. 7 
            
                
                #KNN-RP3Beta-SSLIM 
                w1 = self.knn_rp3_recommender._compute_item_score([user_id_array[i]])
                w1 = w1 / np.linalg.norm(w1, 2)

                w2 = self.sslim_recommender._compute_item_score([user_id_array[i]])
                w2 = w2 / np.linalg.norm(w2, 2)
                item_weights[i,:] = w1*self.peso_1+w2*self.peso_2

                
                """
                RP3Beta-KNN
                w1 = self.rp3beta_recommender._compute_item_score([user_id_array[i]])
                w1 = w1 / np.linalg.norm(w1, 2)

                w2 = self.KNN_recommender._compute_item_score([user_id_array[i]])
                w2 = w2 / np.linalg.norm(w2, 2)
                item_weights[i,:] = w1*self.peso_1+w2*self.peso_2
                """
                #w1 = self.knn_rp3_recommender._compute_item_score([user_id_array[i]])
                #item_weights[i,:] = w1
                
                

            elif interactions >= 1:
            
                w1 = self.sslim_recommender._compute_item_score([user_id_array[i]])
                item_weights[i,:] = w1 

            else:
                print("toppop")
                w1 = self.most_viewed._compute_item_score([user_id_array[i]])
                item_weights[i,:] = w1 

        return item_weights
