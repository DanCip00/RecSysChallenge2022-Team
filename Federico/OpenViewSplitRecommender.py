################################# IMPORT RECOMMENDERS #################################

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

################################## IMPORT LIBRARIES ##################################

import pandas as pd
from numpy import linalg as LA
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np


#################################### HYBRID CLASS ####################################

class OpenViewSplitRecommender(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "OpenViewSplitRecommender"


    W_sparse = []
    shape = (0,0)
    urm_views: sps.csr_matrix
    urm_opens: sps.csr_matrix
    urm_all: sps.csr_matrix

    lambda1 = 0.9
    lambda2 = 0.1

    def __init__(self, interactions: pd.DataFrame, ICM_type, shape):
        self.shape = shape
        self.load_urm_views(interactions)
        self.load_urm_opens(interactions)
        #self.load_urm_impressions(interactions)
        self.load_urm_all(interactions)
        self.ICM_type = ICM_type

        super().__init__(self.urm_all)

    def fit(self, topK_views = 50, topK_opens = 50, shrink_views = 25, shrink_opens = 25):
        print(f"fit: [{topK_views}, {topK_opens}, {shrink_views}, {shrink_opens}]")

        self.open_recommender = ItemKNNCFRecommender(self.urm_opens)
        self.view_recommender = ItemKNNCFRecommender(self.urm_views)
        self.rp3beta_recommender = RP3betaRecommender(self.urm_views)
        #self.impressions_recommender = ItemKNNCFRecommender(self.urm_impressions)
        #self.easr_recommender = EASE_R_Recommender(self.urm_views)
        self.most_viewed = TopPop(self.urm_views)

        self.open_recommender.fit(topK=topK_opens, shrink=shrink_opens)
        self.view_recommender.fit(topK=topK_views, shrink=shrink_views)
        self.rp3beta_recommender.fit()
        #self.impressions_recommender.fit(topK=50, shrink=25)
        #self.easr_recommender.fit()
        self.most_viewed.fit()


        # Instantiate the recommenders
        #self.recommenders['opens']['itemKNNCF'] = ItemKNNCFRecommender(self.urm_opens)
        #self.recommenders['views']['itemKNNCF'] = ItemKNNCFRecommender(self.urm_views)
        #self.EASE_R = EASE_R_Recommender(self.URM_aug)

        #self.EASE_R.fit()
        # for urm_type in self.recommenders:
        #     for recommender_name in urm_type:
        #         (self.recommenders[urm_type][recommender_name]).fit()

    def _compute_item_score(self,
                            user_id_array,
                            items_to_compute=None
                            ):
        w1 = self.view_recommender._compute_item_score(user_id_array)
        w1 = w1 / np.linalg.norm(w1, 2)

        w2 = self.rp3beta_recommender._compute_item_score(user_id_array)
        w2 = w2 / np.linalg.norm(w2, 2)

        #w_easer = self.easr_recommender._compute_item_score(user_id_array)
        #w_easer = w_easer / np.linalg.norm(w_easer, 2)

        #w12 = 0.5 * w1 + 0.3 * w2 + 0.2 * w_easer
        w12 = 0.4 * w1 + 0.6 * w2

        w3 = self.open_recommender._compute_item_score(user_id_array)
        w3 = w3 / np.linalg.norm(w3, 2)

        #w_impressions = self.impressions_recommender._compute_item_score(user_id_array)
        #w_impressions = w_impressions / np.linalg.norm(w_impressions, 2)

        item_weights = (0.6 * w12) + (0.4 * w3)

        return item_weights

    def get_weights(self, user_id):
        user_weights = self.urm_all[user_id].toarray().ravel()

        view_weights = self.view_recommender._compute_item_score([user_id])
        view_weights /= np.linalg.norm(view_weights, 2)

        opens_weights = self.open_recommender._compute_item_score([user_id])
        opens_weights /= np.linalg.norm(opens_weights, 2)


        top_view_score = self.most_viewed._compute_item_score([user_id])
        top_view_score /= np.linalg.norm(top_view_score, 2)

        w_impressions = self.impressions_recommender._compute_item_score([user_id])
        w_impressions = w_impressions / np.linalg.norm(w_impressions, 2)

        item_weights = (0.6 * view_weights) + (0.4 * opens_weights)

        frame = pd.DataFrame(item_weights.ravel())
        frame.columns = ['item_weights']
        frame['w_impressions'] = w_impressions.ravel()
        frame['view_weights'] = view_weights.ravel()
        frame['open_weights'] = opens_weights.ravel()
        frame['user_weights'] = user_weights
        frame['top_view_score'] = top_view_score.ravel()

        return frame


    def load_urm_views(self, interactions):
        view_interactions = interactions[(interactions['view_count'] > 0)]
        self.urm_views = sps.csr_matrix(
            (view_interactions.ratings, (view_interactions.user_id, view_interactions.item_id)),
            shape=self.shape
        )

    def load_urm_opens(self, interactions):
        open_interactions = interactions[(interactions['open_count'] > 0)].copy()
        self.urm_opens = sps.csr_matrix(
            (open_interactions.ratings, (open_interactions.user_id, open_interactions.item_id)),
            shape=self.shape
        )
    def load_urm_impressions(self, interactions):
        open_interactions = interactions[(interactions['impressions_count'] > 0)].copy()
        self.urm_impressions = sps.csr_matrix(
            (open_interactions.ratings, (open_interactions.user_id, open_interactions.item_id)),
            shape=self.shape
        )

    def load_urm_all(self, interactions):
        real_interactions = interactions[interactions['view_count'] + interactions['open_count'] > 0].copy()
        self.urm_all = sps.csr_matrix(
            (real_interactions.ratings, (real_interactions.user_id, real_interactions.item_id)),
            shape=self.shape
        )