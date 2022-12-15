import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp

from typing import Optional


def not_none(value: Optional, default):
    if value is None:
        return default
    return value


class RatingWeights(object):
    view_rating = 0.33
    open_rating = 0.33
    completion_rating = 0.33

    user_weight = 0.5
    item_weight = 0.5
    view_shrink = 50
    open_shrink = 50

    avg_top_k = 3

    def __init__(self, view_rating=0.33, open_rating=0.33, completion_rating=0.33, user_weight=0.5, item_weight=0.5,
                 view_shrink=50, open_shrink=50, average_top_k=3):
        self.view_rating = view_rating
        self.completion_rating = completion_rating
        self.open_rating = open_rating
        self.user_weight = user_weight
        self.item_weight = item_weight
        self.view_shrink = view_shrink
        self.open_shrink = open_shrink
        self.avg_top_k = average_top_k

    def __str__(self):
        return f"view={self.view_rating}, open={self.open_rating}, completion={self.completion_rating}"

    def clone(self, view_rating: Optional = None, open_rating: Optional = None, completion_rating: Optional = None, user_weight: Optional = None, item_weight: Optional = None,
                 view_shrink: Optional = None, open_shrink: Optional = None, average_top_k: Optional = None):
        new_obj = copy.deepcopy(self)
        new_obj.view_rating = not_none(view_rating, new_obj.view_rating)
        new_obj.completion_rating = not_none(completion_rating, new_obj.completion_rating)
        new_obj.open_rating = not_none(open_rating, new_obj.open_rating)
        new_obj.user_weight = not_none(user_weight, new_obj.user_weight)
        new_obj.item_weight = not_none(item_weight, new_obj.item_weight)
        new_obj.view_shrink = not_none(view_shrink, new_obj.view_shrink)
        new_obj.open_shrink = not_none(open_shrink, new_obj.open_shrink)
        new_obj.avg_top_k = not_none(average_top_k, new_obj.avg_top_k)
        return new_obj


class Hyperparameters(object):
    shrink: int
    top_k: int
    rating_weights: RatingWeights
    normalize = True
    similarity = 'cosine'

    def __init__(self, shrink, top_k, rating_weights: RatingWeights, normalize: bool = True,
                 similarity: str = 'cosine'):
        self.shrink = shrink
        self.top_k = top_k
        self.rating_weights = rating_weights
        self.normalize = normalize
        self.similarity = similarity

    def __str__(self):
        return f"shrink={self.shrink}, top_k={self.top_k}, rating_weights=[{self.rating_weights}], normalize={self.normalize}, similarity={self.similarity}\n"

    def clone(self, shrink: Optional = None, top_k: Optional = None, rating_weights: Optional[RatingWeights] = None,
              normalize: Optional[bool] = None, similarity: Optional[str] = None):
        new_obj = copy.deepcopy(self)
        new_obj.shrink = not_none(shrink, new_obj.shrink)
        new_obj.top_k = not_none(top_k, new_obj.top_k)
        new_obj.rating_weights = not_none(rating_weights, new_obj.rating_weights)
        new_obj.normalize = not_none(normalize, new_obj.normalize)
        new_obj.similarity = not_none(similarity, new_obj.similarity)
        return new_obj


class DataProcessor(object):
    urm: sp.csr_matrix = None
    data: pd.DataFrame
    params: Hyperparameters

    def __init__(self, data: pd.DataFrame, params: Hyperparameters, num_users, num_items):
        self.data = data.copy()
        self.params = params
        self.num_users = num_users
        self.num_items = num_items

    def process_data(self):
        #self.flag_user_outliers()
        self.flag_item_outliers()
        self.add_total_interactions()
        self.add_view_average()
        self.add_open_average()
        self.add_completion()
        self.add_completion_avg()
        self.add_ratings()

        # self.data['ratings'] = 1
        #self.data.loc[self.data.view_count <= 0, 'ratings'] = 2.5

        self.urm = self.to_urm()

        return self

    def to_urm(self):
        if self.urm is not None:
            return self.urm

        return sp.csr_matrix(
            (self.data.ratings, (self.data.user_id, self.data.item_id)),
            shape=(self.num_users, self.num_items)
        )

    def add_total_interactions(self):
        interactions = self.data

        interactions['total_interactions_count'] = interactions.view_count + interactions.open_count

        self.data = interactions

    def add_view_average(self):
        interactions = self.data
        items_average_view_count = interactions[(interactions.view_count >= 1) & (interactions.is_valid == 1)] \
            .sort_values('view_count', ascending=False) \
            .groupby('item_id') \
            .head(self.params.rating_weights.avg_top_k) \
            .groupby('item_id') \
            .mean()['view_count'] \
            .reset_index()
        items_average_view_count.columns = ['item_id', 'item_avg_view']
        # items_average_view_count.loc[items_average_view_count.item_avg_view <= 1, 'item_avg_view'] = None

        user_average_view_count = interactions[(interactions.view_count >= 1) & (interactions.is_valid == 1)] \
            .sort_values('view_count', ascending=False) \
            .groupby('user_id') \
            .head(self.params.rating_weights.avg_top_k) \
            .groupby('user_id') \
            .mean()['view_count'] \
            .reset_index()
        user_average_view_count.columns = ['user_id', 'user_avg_view']
        # user_average_view_count.loc[user_average_view_count.user_avg_view <= 1, 'user_avg_view'] = None

        # Add average view_count for items and user

        result = pd.merge(
            left=interactions,
            right=items_average_view_count,
            on=['item_id'],
            how='left'
        )
        result = pd.merge(
            left=result,
            right=user_average_view_count,
            on=['user_id'],
            how='left'
        )

        self.data = result

    def add_open_average(self):
        interactions = self.data
        items_average_open_count = \
            interactions[(interactions.open_count >= 1) & (interactions.is_valid == 1)].sort_values('open_count', ascending=False).groupby(
                'item_id').head(
                self.params.rating_weights.avg_top_k).groupby('item_id').mean()['open_count'].reset_index()
        items_average_open_count.columns = ['item_id', 'item_avg_open']
        # items_average_open_count.loc[items_average_open_count.item_avg_open <= 1, 'item_avg_open'] = None

        user_average_open_count = \
            interactions[(interactions.open_count >= 1) & (interactions.is_valid == 1)].sort_values('open_count', ascending=False).groupby(
                'user_id').head(
                self.params.rating_weights.avg_top_k).groupby('user_id').mean()['open_count'].reset_index()
        user_average_open_count.columns = ['user_id', 'user_avg_open']
        # user_average_open_count.loc[user_average_open_count.user_avg_open <= 1, 'user_avg_open'] = None

        # Add average view_count for items and user

        result = pd.merge(
            left=interactions,
            right=items_average_open_count,
            on=['item_id'],
            how='left'
        )
        result = pd.merge(
            left=result,
            right=user_average_open_count,
            on=['user_id'],
            how='left'
        )

        self.data = result

    def add_completion(self):
        interactions = self.data
        interactions['completion'] = interactions.view_count / interactions.length
        interactions.loc[interactions.length < 1, 'completion'] = 0
        self.data = interactions

    def add_completion_avg(self):
        interactions = self.data
        items_avg = interactions[(interactions.completion > 0) & (interactions.is_valid == 1)].sort_values('completion', ascending=False).groupby(
            'item_id').head(self.params.rating_weights.avg_top_k).groupby('item_id').mean()['completion'].reset_index()
        items_avg.columns = ['item_id', 'item_avg_completion']
        items_avg.loc[items_avg.item_avg_completion <= 0, 'item_avg_completion'] = None

        user_avg = interactions[(interactions.completion > 0) & (interactions.is_valid == 1)].sort_values('completion', ascending=False).groupby(
            'user_id').head(self.params.rating_weights.avg_top_k).groupby('user_id').mean()['completion'].reset_index()
        user_avg.columns = ['user_id', 'user_avg_completion']
        user_avg.loc[user_avg.user_avg_completion <= 0, 'user_avg_completion'] = None

        # Add average view_count for items and user

        result = pd.merge(
            left=interactions,
            right=items_avg,
            on=['item_id'],
            how='left'
        )
        result = pd.merge(
            left=result,
            right=user_avg,
            on=['user_id'],
            how='left'
        )

        self.data = result

    def add_ratings(self):
        interactions = self.data

        interactions['view_ratings'] = self.params.rating_weights.user_weight * (
                    interactions.view_count / (interactions.user_avg_view + self.params.rating_weights.view_shrink)) \
                                       + self.params.rating_weights.item_weight * (interactions.view_count / (
                    interactions.item_avg_view + self.params.rating_weights.view_shrink))

        interactions['open_ratings'] = self.params.rating_weights.user_weight * (
                    interactions.open_count / (interactions.user_avg_open + self.params.rating_weights.open_shrink)) \
                                       + self.params.rating_weights.item_weight * (interactions.open_count / (
                    interactions.item_avg_open + self.params.rating_weights.open_shrink))

        interactions['completion_ratings'] = self.params.rating_weights.user_weight * (
                    interactions.completion / interactions.user_avg_completion) \
                                             + self.params.rating_weights.item_weight * (
                                                         interactions.completion / interactions.item_avg_completion)

        interactions['view_ratings'].fillna(0, inplace=True)
        interactions['open_ratings'].fillna(0, inplace=True)
        interactions['completion_ratings'].fillna(0, inplace=True)

        rating = self.params.rating_weights.view_rating * interactions.view_ratings \
                                  + self.params.rating_weights.open_rating * interactions.open_ratings \
                                  + self.params.rating_weights.completion_rating * interactions.completion_ratings

        interactions['ratings'] = 1 / (1 + np.exp(-rating))
        #interactions['ratings'] = 1

        interactions.loc[interactions.is_valid != 1, 'ratings'] = 0
        interactions.loc[interactions.is_valid != 1, 'view_ratings'] = 0
        interactions.loc[interactions.is_valid != 1, 'open_ratings'] = 0
        interactions.loc[interactions.is_valid != 1, 'completion_ratings'] = 0
        #interactions.loc[(interactions['view_count'] == 0) & (interactions['open_count'] == 0) & (interactions['impressions_count'] > 0), 'ratings'] = -1

        #interactions['ratings'].fillna(0, inplace=True)

        self.data = interactions


    def flag_user_outliers(self):
        interactions = self.data

        # interactions.loc[interactions.groupby("user_id").view_count.transform(lambda x : (x<x.quantile(0.95))&(x>(x.quantile(0.05)))).eq(1), 'is_valid'] = 1
        # interactions.loc[interactions.groupby("user_id").open_count.transform(lambda x : (x<x.quantile(0.95))&(x>(x.quantile(0.05)))).eq(1), 'is_valid'] = 1

        interactions.loc[interactions.groupby("user_id").view_count.transform(lambda x : (x<x.quantile(0.95))).eq(1), 'is_valid'] = 1
        interactions.loc[interactions.groupby("user_id").open_count.transform(lambda x : (x<x.quantile(0.95))).eq(1), 'is_valid'] = 1

        self.data = interactions

    def flag_item_outliers(self):
        interactions = self.data

        interactions.loc[interactions.groupby("item_id").view_count.transform(lambda x : (x<x.quantile(0.95))).eq(1), 'is_valid'] = 1
        interactions.loc[interactions.groupby("item_id").open_count.transform(lambda x : (x<x.quantile(0.95))).eq(1), 'is_valid'] = 1

        self.data = interactions

