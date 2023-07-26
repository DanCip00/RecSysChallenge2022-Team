import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

from New_Splitting_function.CrossKValidator import CrossKValidator

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from BO.bayes_opt import BayesianOptimization
from Base.Evaluation.Evaluator import EvaluatorHoldout
from BO.bayes_opt import SequentialDomainReductionTransformer
from BO.bayes_opt.logger import JSONLogger
from BO.bayes_opt.event import Events
from Notebooks_utils.data_splitter import train_test_holdout
from Recommenders.MatrixFactorization.IALS_implicit_Recommender import IALSRecommender_implicit
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import pandas as pd
import csv
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
import numpy as np 
import scipy.sparse as sps
from tqdm import tqdm



class ItemKNNDanieleRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2, hybrid_low):
        super(ItemKNNDanieleRecommender, self).__init__(URM_train)
        self.W_sparse = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        self.hybrid_low = hybrid_low

    def fit(self, alpha=0.5, interactions_threshold = 18):
        self.alpha = alpha
        self.interactions_threshold =interactions_threshold

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array, items_to_compute)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array, items_to_compute)
        item_weights_3 = self.hybrid_low._compute_item_score(user_id_array, items_to_compute)


        item_weights = item_weights_1
        for i in range(len(user_id_array)):
            num_interactions = len(self.URM_train[user_id_array[i],:].indices)
            if num_interactions <= self.interactions_threshold:
                item_weights[i] = item_weights_3[i]
            item_weights[i] = self.alpha * item_weights_1[i] + (1 - self.alpha) * item_weights_2[i]

        return item_weights











# ---------------------------------------------------------------------------------------------------------
# Loading URM and ICM

import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.MyDataManager as dm


URMv = dm.getURMviews()
URMo = dm.getURMopen()
URM_all = URMv + URMo
URM_all.data = np.ones(len(URM_all.data))

ICMt=dm.getICMt()
ICMl=dm.getICMl()
ICM_all = mm.augmentedICM(ICMt,ICMl)


icm_length = pd.read_csv('Dataset/data_ICM_length.csv', dtype={0: int, 1: int, 2: int})
feature = np.ones(dm.n_items)

for i in icm_length.item_id:
    feature[i] = 0 


########## RP3Beta ##########
recommender_rp3Beta = RP3betaRecommender(sps.vstack([URM_all,ICM_all.T,feature.T]))
print("Fit RP3Beta")
# {'topK': 69, 'alpha': 0.6854042891733674, 'beta': 0.2763471245555947, 'normalize_similarity': True} -> MAP 0.0282084
recommender_rp3Beta.fit(topK=69, alpha=0.6854042891733674,
                            beta=0.2763471245555947, implicit=True)
    

########## SLIM LOW ##########
"""
recommender_elastic_low = SLIMElasticNetRecommender(URM_all)
# {'alpha': 0.002930092866966509, 'l1_ratio': 0.006239337272696024, 'topK': 882} -> MAP 0.0422894
elastic_param = {'alpha': 0.002930092866966509, 'l1_ratio': 0.006239337272696024, 'topK': 882}
print("Fit SLIM")
recommender_elastic_low.fit(**elastic_param)
recommender_elastic_low.save_model("Daniele/Recommenders/Hybrid_fede/saved_models_SUBBB/","slim_low")
"""
recommender_elastic_low = SLIMElasticNetRecommender(URM_all)
recommender_elastic_low.load_model("Daniele/Recommenders/Hybrid_fede/saved_models_SUBBB/")

########### SLIM  ###########
recommender_elastic_icm = SLIMElasticNetRecommender(sps.vstack([URM_all, ICM_all.T,feature.T]))
"""
#{'alpha': 0.0006123679967876883, 'l1_ratio': 0.004881023904378766, 'topK': 836} -> MAP: 0.0404296
elastic_param = {'topK': 836, 'l1_ratio': 0.004881023904378766, 'alpha': 0.0006123679967876883}
print("Fit SLIM")
recommender_elastic_icm.fit(**elastic_param)
recommender_elastic_icm.save_model("Daniele/Recommenders/Hybrid_fede/saved_models_SUBBB/","slim_icm")
"""
recommender_elastic_icm.load_model("Daniele/Recommenders/Hybrid_fede/saved_models_SUBBB/","slim_icm")

########## IALS ##########
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

ials = IALSRecommender(URM_train=sps.vstack([URM_all, ICM_all.T,feature.T]))
ials.load_model("Daniele/Recommenders/Hybrid_fede/saved_models_SUBBB/","ials_fede_01_URM_all")


from Recommenders.Hybrid_recommender.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender
hybrid_low = ItemKNNScoresHybridRecommender(URM_all,Recommender_1=recommender_elastic_low,Recommender_2=ials)
hybrid_low.fit(alpha=0.85)


recommender = ItemKNNDanieleRecommender(URM_train=URM_all,
                                        Recommender_1=recommender_elastic_icm,
                                        Recommender_2=recommender_rp3Beta,
                                        hybrid_low=recommender_elastic_low)


recommender.fit(alpha=0.5198964505232709,interactions_threshold=18.733088096929528)


f = open("submission_lastDance3.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(dm.getUserIDtest_df().user_id):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
    well_formatted = " ".join([str(x) for x in recommended_items])
    f.write(f"{t}, {well_formatted}\n")

