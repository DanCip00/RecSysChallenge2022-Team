import os 
import sys
while os.path.split(os.getcwd())[1] != 'RecSysChallenge2023-Team':
    os.chdir('..')
sys.path.insert(1, os.getcwd())

import Daniele.Utils.MyDataManager as dm
import scipy.sparse as sps
from tqdm import tqdm
import numpy as np
import math 

def _norma_exp(urm, slope, center,verbose=False):
        # Normalization

        if verbose:
            print(urm.min(),urm.max(),urm.data.mean())       

        urm = urm.tocsr().astype("float")
        for i in tqdm(range(len(urm.data))):                                  
            urm.data[i] = 1/(1+math.e**(-urm.data[i]*slope+ center[urm.indices[i]]))+1
        
        if verbose:
            print(urm.min(),urm.max(),urm.data.mean()) 
        return urm 

def _delete_row_lil(mat, i):
    if not isinstance(mat, sps.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def explicitURM(urm,slope=0.01,n_remove=2750,shrink_bias=85,bias='item',new_val=0, verbose=False):
    """
    Hyperparameters:
     - URMv ->  slope=0.01, n_remove=2750, shrink_bias=85,bias='item', new_val = 0
     - URMo ->  slope=0.01, n_remove = 10000, shrink_bias = 25,bias='user', new_val = 30
    """
    urm=urm.copy()

    index = np.flip(np.argsort(urm.data),-1)
    for i in range (n_remove):
        urm.data[index[i]]=new_val

    if bias == 'item':
        urm= urm.tocsc()
        bias = np.zeros(urm.shape[1])
        for item in tqdm(range(urm.shape[1])):
            bias[item] = urm.getcol(item).sum()/(np.count_nonzero(urm.getcol(item).data)+shrink_bias)
        urm = urm.tocsr()
    else:      #user
        bias = np.zeros(urm.shape[0])
        for user in tqdm(range(urm.shape[1])):
            bias[user] = urm.getrow(user).sum()/(np.count_nonzero(urm.getrow(user).data)+shrink_bias)

    urm = _norma_exp(urm,slope,bias,verbose)
    
    return urm 


def augmentURM(urm,min_profile_length=20,max_profile_length=100, keep_interactions=0.66):
    """
    Hyperparameters:
    - min_profile_length=20
    - max_profile_length=100
    - keep_interactions=0.66
    """
    urma = urm.copy().tolil()

    for i in reversed(range(urm.shape[0])):
        row = urma.getrow(i)
        if row.nnz < min_profile_length or row.nnz > max_profile_length:
            _delete_row_lil(urma,i)

    urma = urma.tocoo()
    n_interactions =  urma.nnz
    aug_mask = np.random.choice([True,False], n_interactions, p=[keep_interactions, 1-keep_interactions])

    return  sps.csr_matrix((urma.data[aug_mask],
                            (urma.row[aug_mask], urma.col[aug_mask])),shape=urma.shape)


def augmentedICM(ICMt, ICMl, max_length=50):

    icml = ICMl.copy()
    
    icml= icml.tocoo()
    for i in range (len(icml.data)):
        if icml.col[i]>max_length:
            icml.col[i]=max_length
    icml=icml.tocsr()

    is_series = np.zeros(ICMt.shape[0],dtype=int)
    is_series[icml.tocsc().indices] = 1

    is_film = np.ones(ICMt.shape[0],dtype=int)
    is_film-=is_series

    is_series = sps.csr_matrix(is_series)
    is_film = sps.csr_matrix(is_film)

    a1 = icml.getcol(0)
    for index in range(1,5):
        a1 +=icml.getcol(index)

    a2 = icml.getcol(5)
    for index in range(6,15):
        a2 +=icml.getcol(index)

    a3 = icml.getcol(15)
    for index in range(16,30):
        a3 +=icml.getcol(index)


    a4 = icml.getcol(30)
    for index in range(31,icml.indices.max()):
        a4 +=icml.getcol(index)

    icm = ICMt.copy()
    return sps.hstack([icm,is_series.T,is_film.T,a1,a2,a3,a4]).tocsr()
    

def defaultWrap(urmv, urmo, icmt=None, icml=None, beta=1,add_aug=True,appendICM=False):

    urmv = explicitURM(urmv,slope=0.01, n_remove=2750, shrink_bias=85,bias='item', new_val = 0)

    urmo = explicitURM(urmo, slope=0.01, n_remove = 10000, shrink_bias = 25,bias='user', new_val = 30)

    ##### Unione delle 2 matrici #####
    URMc = urmv.multiply(urmo)
    URMc.data = np.ones(len(URMc.data))

    urmo -=URMc.multiply(urmo)
    urm = urmv+urmo*beta     
    ##################################

    if add_aug:
        urma = augmentURM(urm,min_profile_length=20,max_profile_length=100, keep_interactions=0.66)
        urm = sps.vstack([urm,urma]) 
    
    if appendICM:
        icma = augmentedICM(icmt,icml)
        urm = sps.vstack([urm,icma.T])

    return urm 



        
