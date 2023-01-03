import sys
import numpy as np
import pandas as pd
from tqdm import tqdm  # for a nice progress bar
from matplotlib import pyplot as plt
from sklearn.preprocessing import (
    # very good class for handling 1-hot feature encoding (especially for unseen categories in test data etc.)
    OneHotEncoder,
)
import warnings
from pprint import pprint

import torch
from torch import nn
from torch.utils.data import DataLoader

## simulate user/item data ## ---------------------------------------------------------------------
sys.path.append("..")
from recsys_simulation import recsys_data_simulator

"""
I simulate implicit ratings data:
    I expose each simulated user to every item in the simulated item catalogue..
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

sim_n_users = 1_000  # 10_000
sim_n_items = 50  # 200

sim_obj = recsys_data_simulator(
    n_users=sim_n_users,
    n_items=sim_n_items,
    n_user_types=10,
    n_item_attr_pref_mutations_per_user=5,
    n_additional_context_modifiers_per_user=5,
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
        "max": 0.50,
    },  # I use this "rating" as "item purchase probability"
    rating_trunc_norm_std_dev=0.01,
    n_context_effects=5,
    context_effect_abs_size=5,
)

# expose each user to the entire item catalogue:
# i.e. get the "item purchase probability" for each user/item combination
# (each exposure in a random context)
sim_obj.expose_each_user_to_k_items(
    min_k=sim_n_items,
    max_k=sim_n_items,
    ignore_context=False,
    add_noise_to_rating=False,  # this speeds up the function (and makes the patterns easier to model)
)

# create pandas dataframes containing the population of users and items:
# (and their attributes)
# (for quick lookup)
user_attr_df = sim_obj.user_attr_data_to_pandas_df()
item_attr_df = sim_obj.item_attr_data_to_pandas_df()

# collect training examples in a pandas data-frame:
temp_df_row_list = []
item_id_set = set(sim_obj.item_dict.keys())
for user_id in tqdm(sim_obj.user_dict, desc="creating training df"):
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
raw_train_df["example_ID"] = range(len(raw_train_df))  # unique row (sample) identifier
raw_train_df.set_index(
    # this makes joining on these columns a lot faster
    ["example_ID", "user_ID", "item_ID"],
    inplace=True,
)

# make some continuous features based on the context features and add them to the simulated data
# This simply adds redundant predictive information..
# ..but I include it in order to show the full functionality of the model
for context_x in sim_obj.potential_context_attr:
    context_vals = sim_obj.potential_context_attr[context_x]
    val_range_len = 200 / len(context_vals)
    val_range_ref = {}
    for i in range(len(context_vals)):
        val_range_ref[context_vals[i]] = -100 + i * val_range_len
    raw_train_df[f"{context_x}_continuous"] = np.array(
        [val_range_ref[x] for x in raw_train_df[context_x].values]
    ) + np.random.uniform(low=0, high=val_range_len, size=len(raw_train_df))

## each user contributes 1 observation to validation set and the rest to the training set:
# i.e. every user has observations in both training and validation
#      (must have this in order to validate the user embeddings)
#  users with only 1 observation are put into the training set
pd_df_list = []
for user_id in tqdm(
    raw_train_df.reset_index()["user_ID"].drop_duplicates(), desc="train/valid split"
):
    user_info = raw_train_df.loc[:, user_id, :][[]].reset_index()
    user_info["user_ID"] = user_id
    user_info.set_index(["example_ID", "user_ID", "item_ID"], inplace=True)
    if len(user_info) == 1:
        user_info["model_data_partition"] = "train"
    else:
        user_info["model_data_partition"] = np.random.choice(
            ["validate"] + ["train"] * (len(user_info) - 1),
            size=len(user_info),
            replace=False,
        )
    pd_df_list.append(user_info.copy())
model_data_partition_ref_df = pd.concat(pd_df_list, axis=0)

model_data_df = raw_train_df.join(model_data_partition_ref_df)

# add the train/valid partition into the multi-index:
model_data_df = model_data_df.reset_index().set_index(
    ["example_ID", "user_ID", "item_ID", "model_data_partition"]
)

## Deep & Cross Model - Implemented in PyTorch ## -----------------------------------------------
"""
-- Deep & Cross Model References --
1. "Wide & Deep Learning for Recommender Systems" (https://arxiv.org/abs/1606.07792)
2. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (https://arxiv.org/abs/2008.13535)
"""

pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch using device: '{pytorch_device}'")

input_features_control_dict = {
    # this dictionary tells the model which features to include in the model, and exactly what to do with them
    # each feature will feed into one (or more) of the following model components:
    #       1. "direct_to_deep":            (for continuous features): Feature fed straight into the deep part of the model, without any preprocessing
    #       2. "direct_to_cross":           (for continuous features): Feature fed straight into the cross part of the model, without any preprocessing
    #       3. "one_hot_then_deep":         (for categorical features): Feature 1-hot encoded then fed directly into the deep part of the model
    #       4. "embed_then_deep":           (for categorical features): Feature embedded then fed into the deep part of the model
    #       5. "one_hot_then_cross":        (for categorical features): Feature 1-hot encoded then fed into the cross (wide) part of the model
    #       6. "embed_then_cross":          (for categorical features): Feature embedded then fed into the cross (wide) part of the model    
        "user_ID":{
            "category":"ID_field",
            "send_to":["embed_then_deep"]
        },
        "item_ID":{
            "category":"ID_field",
            "send_to":["embed_then_deep"]
        },
        "location": {
            "category": "user_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "age_group":{
            "category": "user_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },  
        "affluence":{
            "category": "user_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "main_device":{
            "category": "user_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "colour":{
            "category": "item_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "size":{
            "category": "item_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "material":{
            "category": "item_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "style":{
            "category": "item_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "time_of_day_continuous":{
            "category": "context_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "day_of_week_continuous":{
            "category": "context_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "social_context_continuous":{
            "category": "context_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
        "user_group_recommendation_continuous":{
            "category": "context_attribute",
            "send_to": ["one_hot_then_deep", "one_hot_then_cross"]
        },
}

include_featureNames_dict = {
    # this dictionary is used to specify which features are included in the model..
    # ...and also how each feature is used in the model
    # each feature will feed into one (or more) of the following model components:
    #
    #       1. "direct_to_deep":            (for continuous features): Feature fed straight into the deep part of the model, without any preprocessing
    #       2. "direct_to_cross":           (for continuous features): Feature fed straight into the cross part of the model, without any preprocessing
    #       3. "one_hot_then_deep":         (for categorical features): Feature 1-hot encoded then fed directly into the deep part of the model
    #       4. "embed_then_deep":           (for categorical features): Feature embedded then fed into the deep part of the model
    #       5. "one_hot_then_cross":        (for categorical features): Feature 1-hot encoded then fed into the cross (wide) part of the model
    #       6. "embed_then_cross":          (for categorical features): Feature embedded then fed into the cross (wide) part of the model
    #
    # NOTE: the same feature can be included in more than one part of the model (e.g. "direct_to_deep" and "direct_to_cross")
    "direct_to_deep": [
        "time_of_day_continuous",  # recommendation context feature
        "day_of_week_continuous",  # recommendation context feature
        "social_context_continuous",  # recommendation context feature
        "user_group_recommendation_continuous",  # recommendation context feature
    ],
    "direct_to_cross": ["time_of_day_continuous"],
    "one_hot_then_deep": [
        "time_of_day",  # recommendation context feature
        "day_of_week",  # recommendation context feature
        "social_context",  # recommendation context feature
        "user_group_recommendation",  # recommendation context feature
        "location",  # user feature
        "age_group",  # user feature
        "affluence",  # user feature
        "main_device",  # user feature
        "colour",  # item feature
        "size",  # item feature
        "material",  # item feature
        "style",  # item feature
    ],
    "embed_then_deep": ["user_ID", "item_ID"],
    "one_hot_then_cross": [
        "time_of_day",  # recommendation context feature
        "day_of_week",  # recommendation context feature
        "social_context",  # recommendation context feature
        "user_group_recommendation",  # recommendation context feature
        "location",  # user feature
        "age_group",  # user feature
        "affluence",  # user feature
        "main_device",  # user feature
        "colour",  # item feature
        "size",  # item feature
        "material",  # item feature
        "style",  # item feature
    ],
    "embed_then_cross": ["time_of_day", "day_of_week"],
}

# embeddings in PyTorch use an integer ID (e.g. n unique labels must use IDs 0,1,2,...,n for the embedding part of the model)
# this dictionary ("feature_embed_idx_ref_dict") is used to map each example from raw data ID (e.g. user_ID, item_ID etc.) into an embedding integer ID (and back again):
feature_embed_idx_ref_dict = {}
featureNames_to_embed = set(
    include_featureNames_dict["embed_then_deep"]
    + include_featureNames_dict["embed_then_cross"]
)
for x_name in featureNames_to_embed:
    feature_embed_idx_ref_dict[x_name] = {}
    if x_name in ("user_ID", "item_ID"):
        unique_vals_list = (
            # all user_ID seen in training data
            model_data_df.loc[:, :, :, "train"]
            .reset_index()[x_name]
            .drop_duplicates()
            .values
        )
        np.random.shuffle(
            unique_vals_list
        )  # wouldn't do this in a real application - just doing it here to illustrate what the code is doing
        feature_embed_idx_ref_dict[x_name]["to_embed_ID"] = {
            unique_vals_list[i]: i for i in range(len(unique_vals_list))
        }
        feature_embed_idx_ref_dict[x_name]["from_embed_ID"] = {
            feature_embed_idx_ref_dict[x_name]["to_embed_ID"][eid]: eid
            for eid in unique_vals_list
        }
    elif x_name in model_data_df.columns:
        unique_vals_list = model_data_df[x_name].drop_duplicates().values
        feature_embed_idx_ref_dict[x_name]["to_embed_ID"] = {
            unique_vals_list[i]: i for i in range(len(unique_vals_list))
        }
        feature_embed_idx_ref_dict[x_name]["from_embed_ID"] = {
            feature_embed_idx_ref_dict[x_name]["to_embed_ID"][val_id]: val_id
            for val_id in unique_vals_list
        }
    elif x_name in user_attr_df.columns:
        unique_vals_list = user_attr_df[x_name].drop_duplicates().values
        feature_embed_idx_ref_dict[x_name]["to_embed_ID"] = {
            unique_vals_list[i]: i for i in range(len(unique_vals_list))
        }
        feature_embed_idx_ref_dict[x_name]["from_embed_ID"] = {
            feature_embed_idx_ref_dict[x_name]["to_embed_ID"][val_id]: val_id
            for val_id in unique_vals_list
        }
    elif x_name in item_attr_df.columns:
        unique_vals_list = item_attr_df[x_name].drop_duplicates().values
        feature_embed_idx_ref_dict[x_name]["to_embed_ID"] = {
            unique_vals_list[i]: i for i in range(len(unique_vals_list))
        }
        feature_embed_idx_ref_dict[x_name]["from_embed_ID"] = {
            feature_embed_idx_ref_dict[x_name]["to_embed_ID"][val_id]: val_id
            for val_id in unique_vals_list
        }
    else:
        warnings.warn(f"feature '{x_name}' not found")
        del feature_embed_idx_ref_dict[x_name]

pprint(feature_embed_idx_ref_dict)

# define model parameters:
# (in a real setting, these would likely be learned using hyper-parameter search or optimisation method)
model_hyperParams_dict = {
    "learning_rate": 0.0001,
    "embed_dim": "TODO",  # should this be different per variable?
}

# set up the 1-hot encoder classes which will be used to 1-hot encode the categorical features:
# (there is a separate encoder stored for each feature)
one_hot_encoders_dict = {}
featureNames_to_one_hot = set(
    include_featureNames_dict["one_hot_then_deep"]
    + include_featureNames_dict["one_hot_then_cross"]
)
for x_name in featureNames_to_one_hot:
    one_hot_encoders_dict[x_name] = OneHotEncoder(
        sparse_output=False,  # Will return sparse matrix if set True else will return an array
        handle_unknown="ignore",  # When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros
        dtype=np.float32,  # Desired data type of output
    )
    if x_name in ("user_ID", "item_ID"):
        one_hot_encoders_dict[x_name].fit(
            (
                model_data_df.loc[:, :, :, "train"]
                .reset_index()[x_name]
                .drop_duplicates()
                .to_numpy()
                .reshape(-1, 1)
            )
        )
    elif x_name in model_data_df.columns:
        one_hot_encoders_dict[x_name].fit(
            (model_data_df.loc[:, :, :, "train"][x_name].to_numpy().reshape(-1, 1))
        )
    elif x_name in user_attr_df.columns:
        one_hot_encoders_dict[x_name].fit(
            (user_attr_df[x_name].to_numpy().reshape(-1, 1))
        )
    elif x_name in item_attr_df.columns:
        one_hot_encoders_dict[x_name].fit(
            (item_attr_df[x_name].to_numpy().reshape(-1, 1))
        )
    else:
        warnings.warn(f"feature '{x_name}' not found")
        del one_hot_encoders_dict[x_name]

# put data into a PyTorch-friendly format:
class dcn_model_dataset_class(torch.utils.data.Dataset):
    """A torch dataset class is used to store data in a format that PyTorch is designed to interact with"""

    def __init__(
        self,
        transaction_df,
        user_attribute_df,
        item_attribute_df,
        y_vec,
        input_feature_control_dict,
    ):
        """
        TODO: proper documentation here
        """
        super(
            dcn_model_dataset_class, self
        ).__init__()  # this allows us to inherit from the parent class (torch.utils.data.Dataset)
        self.X_df = X_df
        self.y_vec = y_vec

    def __len__(self):
        """
        TODO: proper documentation here
        """
        return len(self.y_vec)

    def __getitem__(self, idx):
        """
        TODO: proper documentation here
        """
        X_vec = self.X_matrix[idx, :]
        y_scalar = self.y_vec[idx]

        return (
            torch.tensor(X_vec, dtype=torch.int),
            torch.tensor(y_scalar, dtype=torch.int),
        )


dcn_model_train_data = dcn_model_dataset_class(
    transaction_df = model_data_df.loc[:,:,:,"train"],
    user_attribute_df,
    item_attribute_df,
    y_vec,
    input_feature_control_dict=include_featureNames_dict,
)
