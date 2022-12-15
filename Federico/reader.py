import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from Federico.recommender import Hyperparameters, RatingWeights, DataProcessor


class Reader(object):
    data_processor: DataProcessor = None
    interactions: pd.DataFrame = None
    num_users = 0
    num_items = 0

    def __init__(self):
        self.read_data()
        self.get_size()

    def read_data(self):
        interactions_and_impressions = pd.read_csv(
            'Dataset/interactions_and_impressions.csv',
            header=0,
            dtype={0:int, 1:int, 2:str, 3:bool}
        )
        interactions_and_impressions.columns = ['user_id', 'item_id', 'impressions', 'data']
        icm_length = pd.read_csv('Dataset/data_ICM_length.csv')
        icm_type = pd.read_csv('Dataset/data_ICM_type.csv')

        ############### transform interactions into ratings ###############
        view_ratings = interactions_and_impressions[interactions_and_impressions['data'] == 0] \
            .groupby(['item_id', 'user_id']).size().to_frame().reset_index()
        view_ratings.columns = ['item_id', 'user_id', 'view_count']

        open_ratings = interactions_and_impressions[interactions_and_impressions['data'] == 1] \
            .groupby(['item_id', 'user_id']).size().to_frame().reset_index()
        open_ratings.columns = ['item_id', 'user_id', 'open_count']

        interactions = pd.merge(
            left=interactions_and_impressions.groupby('user_id').size().to_frame().reset_index()['user_id'],
            right=view_ratings,
            how='left',
            on=['user_id']
        )

        interactions = pd.merge(
            left=interactions,
            right=open_ratings,
            how='outer',
            on=['item_id', 'user_id'],
        )
        interactions.columns = ['user_id', 'item_id', 'view_count', 'open_count']

        interactions = pd.merge(
            left=interactions,
            right=icm_length[['item_id', 'data']],
            on=['item_id'],
            how='left'
        )

        interactions.columns = ['user_id', 'item_id', 'view_count', 'open_count', 'length']

        interactions = pd.merge(
            left=interactions,
            right=icm_type[['item_id', 'feature_id']],
            on=['item_id'],
            how='left'
        )

        interactions.columns = ['user_id', 'item_id', 'view_count', 'open_count', 'length', 'type']

        interactions['view_count'].fillna(0, inplace=True)
        interactions['open_count'].fillna(0, inplace=True)

        self.interactions = interactions

    def get_size(self):
        interactions = self.interactions
        unique_users = interactions.user_id.unique()
        unique_items = interactions.item_id.unique()

        num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
        num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

        print(num_users, min_user_id, max_user_id)
        print(num_items, min_item_id, max_item_id)
        print(len(interactions))

        self.num_users = num_users
        self.num_items = num_items

        return num_users, num_items

    def get_train_test_split(self, testing_percentage = 0.15):
        seed = 1234
        return train_test_split(
            self.interactions,
            test_size=testing_percentage,
            shuffle=True,
            random_state=seed
        )

    def process_data(self, data: pd.DataFrame):
        self.data_processor = DataProcessor(
            data=data,
            params=self.get_hyperparams(),
            num_users=self.num_users,
            num_items=self.num_items
        )

        return self.data_processor.process_data().data

    def get_hyperparams(self):
        return Hyperparameters(
            shrink=25,
            top_k=50,
            rating_weights=RatingWeights(
                view_rating=0.60,
                open_rating=0.10,
                completion_rating=0.30,
                item_weight=0.1,
                user_weight=0.9,
                open_shrink=25,
                view_shrink=0,
                average_top_k=3
            )
        )

    def to_urm(self, data: pd.DataFrame):
        data['ratings'] = 1
        return sp.csr_matrix(
            (data.ratings, (data.user_id, data.item_id)),
            shape=(self.num_users, self.num_items)
        )


