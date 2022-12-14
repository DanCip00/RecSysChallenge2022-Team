# -*- coding: utf-8 -*-
"""

Module for importing directly the sparces matrices 

"""
import os 
os.chdir("../..")

import pandas as pd
import scipy.sparse as sps
import Daniele.Utils.SaveSparceMatrix as ssm
import numpy as np
    

n_users = 41629 
n_items = 27968
n_features = 7


userIDtest_df = None
URM_views = None
URM_open = None
ICMt = None
ICMl = None

dir_dataset = os.path.join(os.getcwd(),'Dataset')  
dir_matrix = os.path.join(dir_dataset,'matrices')
if not os.path.exists(dir_matrix):
    os.makedirs(dir_matrix, exist_ok=True)


def csvReader(file_path, dtype, usecols):
    return pd.read_csv(filepath_or_buffer=file_path,                      
                       header=0, 
                       usecols=usecols,
                       dtype=dtype,
                       engine='python')


def __saveURMs():
    global URM_views, URM_open

    if os.path.exists(dir_matrix+"/URM_views_coo.csv") and os.path.exists(dir_matrix+"/URM_open_coo.csv") :

        URM_views = ssm.readMatrix(dir_matrix+"/URM_views_coo.csv")
        URM_open = ssm.readMatrix(os.path.join(dir_matrix,'URM_open_coo.csv'))
    
    else:
        ######### Read CSV #########
        II_df = csvReader(os.path.join(dir_dataset,'interactions_and_impressions.csv'),{0:int, 1:int, 2:object, 3:int},[0,1,2,3])

        # Creating sparse matrices #
        URM_views_df = II_df[II_df.Data==0].drop(['Impressions', 'Data'],axis=1).groupby(['UserID','ItemID'], as_index=False,group_keys=False).size()
        URM_views_coo = sps.coo_matrix((URM_views_df["size"].values,
                                    (URM_views_df["UserID"].values,URM_views_df["ItemID"].values)),
                                    shape=(n_users,n_items))

        URM_open_df = II_df[II_df.Data==1].drop(['Impressions', 'Data'],axis=1).groupby(['UserID','ItemID'], as_index=False,group_keys=False).size()
        URM_open_coo = sps.coo_matrix((URM_open_df["size"].values,
                                      (URM_open_df["UserID"].values,URM_open_df["ItemID"].values)),
                                      shape=(n_users,n_items))

        URM_open = URM_open_coo.tocsr()
        URM_views = URM_views_coo.tocsr()
        ######## Save CSV ##########

        ssm.saveMatrix(os.path.join(dir_matrix,'URM_views_coo.csv'),URM_views)
        ssm.saveMatrix(os.path.join(dir_matrix,'URM_open_coo.csv'),URM_open)

        URM_views_coo = URM_open_coo  = II_df = None


def __saveICMt():
    global ICMt

    if not os.path.isfile(os.path.join(dir_matrix,'ICMt.csv')):

        ######### Read CSV #########
        ICM_type_df = csvReader(os.path.join(dir_dataset,'data_ICM_type.csv'),{0:int, 1:int},[0,1])
        
        ##### Mapping Features #####
        mapped_featureID, original_featureID = pd.factorize(ICM_type_df.feature_id.unique())
        featureID_map = pd.Series(mapped_featureID,index=original_featureID)
        ICM_type_df['feature_id']=ICM_type_df['feature_id'].map(featureID_map)

        # Creating sparse matrices #
        ICM_coo = sps.coo_matrix((np.ones(len(ICM_type_df), dtype=int),
                                (ICM_type_df['item_id'].values,ICM_type_df['feature_id'].values)),
                                shape=(n_items,n_features))
        ICMt = ICM_coo.tocsr()

        ssm.saveMatrix(os.path.join(dir_matrix,'ICMt.csv'),ICMt)

        ICM_type_df = ICM_coo = None
    else:
        ICMt = ssm.readMatrix(os.path.join(dir_matrix,'ICMt.csv'))

def __saveICMl():
    global ICMl

    if not os.path.isfile(os.path.join(dir_matrix,'ICMl.csv')):

        ######### Read CSV #########
        ICM_length_df = csvReader(dir_dataset+'/data_ICM_length.csv',{0:int, 1:int},[0,2])
        
        
        # Creating sparse matrices #
        ICM_coo = sps.coo_matrix((np.ones(len(ICM_length_df[ICM_length_df.data!=1]), dtype=int),
                                (ICM_length_df[ICM_length_df.data!=1]['item_id'].values,ICM_length_df[ICM_length_df.data!=1]['data'].values)),
                                shape=(n_items,ICM_length_df.data.max()+1))
        ICMl = ICM_coo.tocsr()

        ssm.saveMatrix(os.path.join(dir_matrix,'ICMl.csv'),ICMl)

        ICM_length_df = ICM_coo = None
    else:
        ICMl = ssm.readMatrix(os.path.join(dir_matrix,'ICMl.csv'))




def getURMviews():
    global URM_views
    if URM_views == None :
        __saveURMs()
    return URM_views

def getURMopen():
    global URM_open
    if URM_open == None :
        __saveURMs()
    return URM_open

def getICMt():
    global ICMt
    if ICMt == None :
        __saveICMt()
    return ICMt

def getICMl():
    global ICMl
    if ICMl == None :
        __saveICMl()
    return ICMl


def getUserIDtest_df():
    global userIDtest_df 
    if not isinstance(userIDtest_df,pd.DataFrame):
        userIDtest_df = csvReader(dir_dataset+'/data_target_users_test.csv',
                                  {0:int},
                                  [0])
    return userIDtest_df


