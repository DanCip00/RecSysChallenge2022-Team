#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import scipy.sparse as sps
import similaripy
import numpy as np


class KNN_CFCBF_custom(ItemKNNCBFRecommender):
    """ ItemKNN_CFCBF_Hybrid_Recommender"""

    RECOMMENDER_NAME = "KNN_CFCBF_custom"

    def __init__(self, urmv,urmo, ICM_train, verbose = True):
        self.urmv = urmv
        self.urmo = urmo
        super(KNN_CFCBF_custom, self).__init__(urmv+urmo, ICM_train, verbose = verbose)

    def fit(self, ICM_weight = 1.0,beta=1.0, **fit_args):

        print("Beta->",beta,"\tICM_weight->",ICM_weight)

        urmv =self.urmv.copy()
        urmo =self.urmo.copy()

        URMc = urmv.multiply(urmo)
        URMc.data = np.ones(len(URMc.data))

        urmo -=URMc.multiply(urmo)
        self.URM_train=similaripy.normalization.bm25plus(urmv+urmo*beta)

        self.ICM_train = self.ICM_train*ICM_weight
        self.ICM_train = sps.hstack([self.ICM_train, self.URM_train.T], format='csr')

        super(KNN_CFCBF_custom, self).fit(**fit_args)


    def _get_cold_item_mask(self):
        return np.logical_and(self._cold_item_CBF_mask, self._cold_item_mask)

