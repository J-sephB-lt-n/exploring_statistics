## simulate user/item data ## ---------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm  # for a nice progress bar
import copy

sys.path.append("..")
from recsys_simulation import recsys_data_simulator

"""
I simulate implicit ratings data:
    I expose each simulated user to every item in the simulated catalogue..
        ..in order to obtain a buy probability (p) for that user/item combination
    Then, each user purchases each item at random, with probability (p)

In order to train the model:
    * I use the observed buying behaviour as positive examples (y=1)
    * For each positive example (y=1), I create a matching random negative example (y=0)..
        ..by including a random item that the user didn't purchase (in the same context as the positive example)

For simplicity here, users without any item purchases are removed from model training and evaluation
    (since these users complicate the user embedding part of the model)
    (in practice, these users will need to be addressed, either within the model or by a different model)
"""

sim_n_users = 100
sim_n_items = 20

sim_obj = recsys_data_simulator(
    n_users=sim_n_users,
    n_items=sim_n_items,
    n_user_types=10,
    n_item_attr_pref_mutations_per_user=10,
    n_additional_context_modifiers_per_user=2,
    potential_item_attr={
        "colour": [
            "red",
            "green",
            "blue",
            "black",
            "white",
            "purple",
            "yellow",
            "pink",
        ],
        "size": ["small", "medium", "large"],
        "material": ["metal", "wood", "cotton", "plastic", "wool", "stone", "glass"],
        "style": [
            "industrial",
            "warm",
            "loud",
            "gothic",
            "tech",
            "sport",
            "floral",
            "pastel",
            "chic",
            "beach",
        ],
    },
    potential_user_attr={
        "location": [
            "cape_town",
            "london",
            "dubai",
            "new_york",
            "rotterdam",
            "porto",
            "tokyo",
        ],
        "age_group": ["infant", "child", "teenager", "youth", "middle_aged", "elderly"],
        "affluence": ["very_low", "low", "middle", "high", "very_high"],
        "main_device": [
            "laptop",
            "desktop_computer",
            "phone",
            "tablet",
            "postal_service",
        ],
    },
    potential_context_attr={
        "time_of_day": ["morning", "afternoon", "night"],
        "day_of_week": [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ],
        "social_context": [
            "public_space",
            "public_transport",
            "private_space",
            "private_transport",
        ],
        "user_group_recommendation": [
            "user_alone",
            "small_user_group",
            "large_user_group",
        ],
    },
    rating_range={
        "min": 0,
        "max": 0.25,
    },  # I use this "rating" as "item purchase probability"
    rating_trunc_norm_std_dev=0.01,
    n_context_effects=2,
    context_effect_abs_size=5,
)
# expose each user to the entire item catalogue:
# i.e. get the "item purchase probability" for each user/item combination
# (each exposure in a random context)
sim_obj.expose_each_user_to_k_items(
    min_k=sim_n_items,
    max_k=sim_n_items,
)

# create pandas dataframes containing the population of users and items:
# (and their attributes)
# (for quick lookup)
user_attr_df = sim_obj.user_attr_data_to_pandas_df()
item_attr_df = sim_obj.item_attr_data_to_pandas_df()

# collect training examples in a pandas data-frame:
temp_df_row_list = []
item_id_set = set(sim_obj.item_dict.keys())
for user_id in tqdm(sim_obj.user_dict):
    item_id_buy_history_idx = []
    # generate positive training examples:
    for i in range(len(sim_obj.user_dict[user_id]["item_exposure_history"])):
        item = sim_obj.user_dict[user_id]["item_exposure_history"][i]
        buy_prob = item["true_affinity_to_item"]
        if buy_prob > np.random.uniform():
            temp_df_row_list.append(
                pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "user_ID": [user_id],
                                "item_ID": [item["item_ID"]],
                                "bought": 1,
                            }
                        ),
                        pd.DataFrame(item["recommend_context"], index=[0]),
                    ],
                    axis=1,
                )
            )
            item_id_buy_history_idx.append(i)
            item["bought"] = 1
        else:
            item["bought"] = 0

    item_id_buy_history = set(
        [
            sim_obj.user_dict[user_id]["item_exposure_history"][i]["item_ID"]
            for i in item_id_buy_history_idx
        ]
    )
    item_id_not_bought_list = list(item_id_set.difference(item_id_buy_history))
    # generate negative training examples:
    for i in item_id_buy_history_idx:
        temp_df_row_list.append(
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "user_ID": [user_id],
                            "item_ID": np.random.choice(item_id_not_bought_list),
                            "bought": 0,
                        }
                    ),
                    pd.DataFrame(
                        sim_obj.user_dict[user_id]["item_exposure_history"][i][
                            "recommend_context"
                        ],
                        index=[0],
                    ),
                ],
                axis=1,
            )
        )
raw_train_df = pd.concat(temp_df_row_list, axis=0)
del temp_df_row_list
raw_train_df["example_ID"] = range(len(raw_train_df))
raw_train_df.set_index(
    # this makes joining on these columns a lot faster
    ["example_ID", "user_ID", "item_ID"],
    inplace=True,
)

## 2 (3) Tower Model - Implemented in TensorFlow ## -----------------------------------------------
import tensorflow as tf
import keras
from sklearn.preprocessing import (
    # very good class for handling 1-hot feature encoding (especially for unseen categories in test data etc.)
    OneHotEncoder,
)

# state which features will be used by the model:
include_featureNames_dict = {
    "user_feature_names": ["location", "age_group", "affluence", "main_device"],
    "item_feature_names": ["colour", "size", "material", "style"],
    "context_feature_names": [
        "time_of_day",
        "day_of_week",
        "social_context",
        "user_group_recommendation",
    ],
}

# set up the 1-hot encoder classes which will be used to 1-hot encode the features:
# (there is a separate encoder for each feature set: users, items and contexts)
sk_1hotEncoder_user_attr = OneHotEncoder(
    sparse_output=False,  # Will return sparse matrix if set True else will return an array
    handle_unknown="ignore",  # When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros
    dtype=np.float32,  # Desired data type of output
)
sk_1hotEncoder_item_attr = OneHotEncoder(
    sparse_output=False,  # Will return sparse matrix if set True else will return an array
    handle_unknown="ignore",  # When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros
    dtype=np.float32,  # Desired data type of output
)
sk_1hotEncoder_context_attr = OneHotEncoder(
    sparse_output=False,  # Will return sparse matrix if set True else will return an array
    handle_unknown="ignore",  # When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros
    dtype=np.float32,  # Desired data type of output
)
sk_1hotEncoder_user_attr.fit(user_attr_df)
sk_1hotEncoder_item_attr.fit(item_attr_df)
sk_1hotEncoder_context_attr.fit(
    raw_train_df[include_featureNames_dict["context_feature_names"]].drop_duplicates()
)

# since holding the entire training data in memory would involve a lot of redundancy..
# (i.e. having to repeat item and user attribute columns for every example)
# ..we rather create a DataGenerator class
# ..this class will perform the necessary joins when it fetches a batch of data for the model
class twoTower_dataGenerator(keras.utils.Sequence):
    """
    this class is used to generate batches of data for training a 2-tower model

    code references:
            1.  https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
            2.  https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(
        self,
        example_IDs,  # list of unique example identifiers (sample row IDs)
        batch_size,  # number of examples to return per batch
        include_featureNames_dict,  # dictionary explaining which features are which (provide column names)
    ):
        """
        -- Initialization of Class --

        example of include_featureNames_dict:
            include_featureNames_dict = {
                "user_feature_names": ["age_group", "location"],
                "item_feature_names": ["colour", "size", "material"],
                "context_feature_names": ["time_of_day", "day_of_week"],
            }
        """
        self.example_IDs = example_IDs
        self.batch_size = batch_size
        self.include_featureNames_dict = include_featureNames_dict

    def __len__(self):
        "returns the total number of batches in each epoch"
        return int(
            np.floor(len(self.example_IDs) / self.batch_size)
        )  # this ensures each example appears once per epoch

    def __getitem__(self, batch_idx):
        "generates a single batch of data"
        idx_list = range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
        batch_example_ID_list = (
            raw_train_df.loc[idx_list].reset_index()["example_ID"].values
        )
        batch_user_ID_list = raw_train_df.loc[idx_list].reset_index()["user_ID"].values
        batch_item_ID_list = raw_train_df.loc[idx_list].reset_index()["item_ID"].values

        batch_x = (
            raw_train_df.loc[idx_list]
            .join(item_attr_df, on="item_ID")  # add item attributes data
            .join(user_attr_df, on="user_ID")  # add user attributes data
        )
        batch1_hot_user_features = pd.DataFrame(
            sk_1hotEncoder_user_attr.transform(
                batch_x[self.include_featureNames_dict["user_feature_names"]]
            ),
            columns=sk_1hotEncoder_user_attr.get_feature_names_out(),
        )
        batch1_hot_item_features = pd.DataFrame(
            sk_1hotEncoder_item_attr.transform(
                batch_x[self.include_featureNames_dict["item_feature_names"]]
            ),
            columns=sk_1hotEncoder_item_attr.get_feature_names_out(),
        )
        batch1_hot_context_features = pd.DataFrame(
            sk_1hotEncoder_context_attr.transform(
                batch_x[self.include_featureNames_dict["context_feature_names"]]
            ),
            columns=sk_1hotEncoder_context_attr.get_feature_names_out(),
        )

        batch_y = raw_train_df.loc[idx_list]["bought"].values

        return {
            "example_ID_list": batch_example_ID_list,
            "user_ID_list": batch_user_ID_list,
            "item_ID_list": batch_item_ID_list,
            "user_features_matrix": batch1_hot_user_features,
            "item_features_matrix": batch1_hot_item_features,
            "context_features_matrix": batch1_hot_context_features,
            "user_response_list": batch_y,
        }


dataGenObj = twoTower_dataGenerator(
    example_IDs=raw_train_df.reset_index()["example_ID"].values,
    include_featureNames_dict=include_featureNames_dict,
    batch_size=4,
)

# a quick error-testing of the dataGenerator class:
test_batch = dataGenObj.__getitem__(batch_idx=1)

# define the 2-tower model class:
class tf_2tower_model_class(keras.Model):
    """
    reference:  https://keras.io/api/models/model/
    """

    def __init__(
        self,
        n_users,  # number of unique users (used for the user embeddings)
        n_items,  # number of unique items (used for the item embeddings)
        n_layers_deep,  # number of dense layers in each tower (user tower, item tower, context tower)
        n_nodes_per_layer,  # number of nodes per layer within each tower
        embed_dim,  # the dimension of the user and item embeddings
        final_encoding_dim,  # the desired dimension of the final encoding (output of each tower)
    ):
        """
        TODO: method documentation here

        Parameters
        ----------
        n_users
            TODO
        n_items
            TODO
        n_layers_deep
            number of dense layers in each tower (user tower, item tower, context tower)
        n_nodes_per_layer
            number of nodes per layer within each tower
        embed_dim
            TODO
        final_encoding_dim
            TODO
        """
        super().__init__()
        # save the input parameters as attributes of the class instance:
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers_deep = n_layers_deep
        self.n_nodes_per_layer = n_nodes_per_layer
        self.embed_dim = embed_dim
        self.final_encoding_dim = final_encoding_dim

        # set up the embedding layers:
        self.user_embedding = tf.keras.layers.Embedding(
            n_users,
            embed_dim,
            embeddings_initializer="he_normal",
            name="user_embedding",
        )
        self.item_embedding = tf.keras.layers.Embedding(
            n_items,
            embed_dim,
            embeddings_initializer="he_normal",
            name="item_embedding",
        )

        # set up the other model layers:
        self.user_inputs_concat_layer = keras.layers.Concatenate(axis=1)
        self.user_tower_layers = [
            keras.layers.Dense(
                self.n_nodes_per_layer, activation="relu", name=f"userTower_dense{i}"
            )
            for i in range(self.n_layers_deep)
        ]
        self.user_tower_output_layer = keras.layers.Dense(
            self.final_encoding_dim, activation="relu", name="userTower_output"
        )

        self.item_inputs_concat_layer = keras.layers.Concatenate(axis=1)
        self.item_tower_layers = [
            keras.layers.Dense(
                self.n_nodes_per_layer, activation="relu", name=f"itemTower_dense{i}"
            )
            for i in range(self.n_layers_deep)
        ]
        self.item_tower_output_layer = keras.layers.Dense(
            self.final_encoding_dim, activation="relu", name="itemTower_output"
        )

        self.context_tower_layers = [
            keras.layers.Dense(
                self.n_nodes_per_layer, activation="relu", name=f"contextTower_dense{i}"
            )
            for i in range(self.n_layers_deep)
        ]

    def call(self, inputs):
        """
        TODO: method documentation here

        Parameters
        ----------
        inputs
            input data - in the format produced by twoTower_dataGenerator.__getitem__()
        training
            TODO
        """

        ## build user tower ##
        user_embed_vectors = self.user_embedding(inputs["user_ID_list"])
        user_attr = inputs["user_features_matrix"].to_numpy()
        ux = self.user_inputs_concat_layer([user_embed_vectors, user_attr])
        for i in range(len(self.user_tower_layers)):
            ux = self.user_tower_layers[i](ux)
        user_tower_output = self.user_tower_output_layer(ux)

        ## build item tower ##
        item_embed_vectors = self.item_embedding(inputs["item_ID_list"])
        item_attr = inputs["item_features_matrix"].to_numpy()
        ix = self.item_inputs_concat_layer([item_embed_vectors, item_attr])
        for i in range(len(self.item_tower_layers)):
            ix = self.item_tower_layers[i](ix)
        item_tower_output = self.item_tower_output_layer(ix)

        # element-wise product of the towers:
        elementwise_prod = user_tower_output * item_tower_output

        # sum the element-wise products of the towers:
        dot_prods = tf.reduce_sum(elementwise_prod, axis=1)

        # force the output to the range [0,1] using a sigmoid function:
        return tf.math.sigmoid(dot_prods)


class tf_2tower_model_class(keras.Model):
    """
    reference:  https://keras.io/api/models/model/
    """

    def __init__(
        self,
        n_users,  # number of unique users (used for the user embeddings)
        n_items,  # number of unique items (used for the item embeddings)
        n_layers_deep,  # number of dense layers in each tower (user tower, item tower, context tower)
        n_nodes_per_layer,  # number of nodes per layer within each tower
        embed_dim,  # the dimension of the user and item embeddings
        final_encoding_dim,  # the desired dimension of the final encoding (output of each tower)
    ):
        super().__init__()
        # save the input parameters as attributes of the class instance:
        self.n_users = n_users
        self.n_items = n_items
        # set up the embedding layers:
        self.user_embedding = tf.keras.layers.Embedding(
            n_users,
            embed_dim,
            embeddings_initializer="he_normal",
            name="user_embedding",
        )
        self.item_embedding = tf.keras.layers.Embedding(
            n_items,
            embed_dim,
            embeddings_initializer="he_normal",
            name="item_embedding",
        )

    def call(self, inputs):
        """
        TODO: method documentation here

        Parameters
        ----------
        inputs
            input data - in the format produced by twoTower_dataGenerator.__getitem__()
        training
            TODO
        """

        ## build user tower ##
        user_embed_vectors = self.user_embedding(inputs["user_ID_list"])

        ## build item tower ##
        item_embed_vectors = self.item_embedding(inputs["item_ID_list"])

        # element-wise product of the towers:
        elementwise_prod = user_embed_vectors * item_embed_vectors

        # sum the element-wise products of the towers:
        dot_prods = tf.reduce_sum(elementwise_prod, axis=1)

        # force the output to the range [0,1] using a sigmoid function:
        return tf.math.sigmoid(dot_prods)


tf_2tower_model = tf_2tower_model_class(
    n_users=sim_n_users,  # number of unique users (used for the user embeddings)
    n_items=sim_n_items,  # number of unique items (used for the item embeddings)
    n_layers_deep=4,  # number of dense layers in each tower (user tower, item tower, context tower)
    n_nodes_per_layer=20,  # number of nodes per layer within each tower
    embed_dim=10,  # the dimension of the user and item embeddings
    final_encoding_dim=5,  # # the desired dimension of the final encoding (output of each tower)
)

tf_2tower_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    # stop training there is no improvement in validation loss for x epochs
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)
train_history = tf_2tower_model.fit(
    dataGenObj,
    epochs=5,
    verbose=1,
    callbacks=[early_stopping_callback],
)
