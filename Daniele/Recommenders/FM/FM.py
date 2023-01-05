import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

import numpy as np 


path_save= "Daniele/Recommenders/FM/saved_models"
if not os.path.exists(path_save):
    os.makedirs(path_save)


from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout

import Daniele.Utils.MyDataManager as dm 
import Daniele.Utils.MatrixManipulation as mm
import Daniele.Utils.SaveSparceMatrix as ssm
import scipy.sparse as sps

URMv = dm.getURMviews()
URMo = dm.getURMopen()
ICMt=dm.getICMt()
ICMl=dm.getICMl()

name="train.csv"
dir = os.path.join(path_save,name)
if not os.path.exists(dir):
    URMv_train, URMv_test = split_train_in_two_percentage_global_sample(URMv, train_percentage = 0.80)

    ssm.saveMatrix(dir,URMv_train)

    name="test.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,URMv_test)

    urm_def = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo,icml=ICMl,icmt=ICMt, normalize=True, add_aug=True,appendICM=False)
    name="urm_def.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,urm_def)

    urm_bin = mm.defaultExplicitURM(urmv=URMv_train,urmo=URMo, normalize=False, add_aug=True)
    urm_bin.data = np.ones(len(urm_bin.data))
    name="urm_bin.csv"
    dir = os.path.join(path_save,name)
    ssm.saveMatrix(dir,urm_bin)
    
else:
    URMv_train=ssm.readMatrix(dir)

    name="test.csv"
    dir = os.path.join(path_save,name)
    URMv_test=ssm.readMatrix(dir)

    name="urm_def.csv"
    dir = os.path.join(path_save,name)
    urm_def = ssm.readMatrix(dir)

    name="urm_bin.csv"
    dir = os.path.join(path_save,name)
    urm_bin = ssm.readMatrix(dir)

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_test = EvaluatorHoldout(URMv_test, cutoff_list=[10])


from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
name="slim_elastic_high"
dir = os.path.join(path_save,name)

slim_elastic_high = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=urm_bin)
if not os.path.exists(dir+".zip"):
    
    # {'alpha': 0.002930092866966509, 'l1_ratio': 0.006239337272696024, 'topK': 882} -> MAP 0.0422894
    slim_elastic_high.fit(alpha=0.002930092866966509, l1_ratio=0.006239337272696024, topK=882)
    slim_elastic_high.save_model(path_save,name)
else:
    slim_elastic_high.load_model(path_save,name)

r_slim = slim_elastic_high._compute_item_score(range(dm.n_users))
r_slim = sps.coo_matrix(r_slim)


from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

rp3beta_high = RP3betaRecommender(urm_bin)
# {'topK': 91, 'alpha': 0.7758215673815734, 'beta': 0.2719143753442684, 'normalize_similarity': True} -> MAP 0.0.0273508
rp3beta_high.fit( topK=91, alpha=0.7758215673815734, beta=0.2719143753442684, normalize_similarity=True )

r_rp3beta = rp3beta_high._compute_item_score(range(dm.n_users))
r_rp3beta = sps.coo_matrix(r_rp3beta)


from Recommenders.BaseRecommender import BaseRecommender
from lightfm import LightFM
import numpy as np
from  tqdm import tqdm

class LightFMCFRecommender(BaseRecommender):
    """LightFMCFRecommender"""

    RECOMMENDER_NAME = "LightFMCFRecommender"

    def __init__(self, URM_train,user_features = None):
        self.user_features = user_features
        super(LightFMCFRecommender, self).__init__(URM_train)


    def fit(self, epochs = 300, alpha = 1e-6, n_factors = 10, n_threads = 4):
        
        # Let's fit a WARP model
        self.lightFM_model = LightFM(loss='warp',    # warp
                                     item_alpha=alpha,
                                     no_components=n_factors)
        batch_size = 5
        best_map=-1
        best_epoch = 0 
        for i in tqdm(range (1,int(epochs/batch_size)+1)):
            print("Epochs->",batch_size*i)
            self.lightFM_model = self.lightFM_model.fit_partial(self.URM_train, 
                                        epochs=i*batch_size,
                                        user_features = self.user_features,
                                        num_threads=n_threads)
            result_df, _ = evaluator_test.evaluateRecommender(self)
            print("Iter ",i,": Epochs->",batch_size*i,"\tMAP ->",result_df["MAP"].values[0])
            if result_df["MAP"].values[0] > best_map : 
                best_map = result_df["MAP"].values[0]
                best_epoch = i * batch_size
        print("Best MAP -> ",best_map,"\t Best epoch -> ",best_epoch)
        

                                       
    def _compute_item_score(self, user_id_array, items_to_compute = None):
        
        # Create a single (n_items, ) array with the item score, then copy it for every user
        items_to_compute = np.arange(self.n_items)
        
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index, user_id in enumerate(user_id_array):
            item_scores[user_index] = self.lightFM_model.predict(int(user_id), 
                                                                 items_to_compute)

        return item_scores

user_popularity = np.ediff1d(sps.csr_matrix(urm_def).indptr)
sort = np.argsort(user_popularity)
u = sps.coo_matrix(user_popularity)

recommender = LightFMCFRecommender(urm_def,sps.hstack([r_slim,r_rp3beta,u.T]))
recommender.fit()

result_df, _ = evaluator_test.evaluateRecommender(recommender)
result_df