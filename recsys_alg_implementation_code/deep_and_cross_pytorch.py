"""
-- High Level Script Description --

This script fits a "Deep & Cross Network" (DCN) model to a simulated recommendation dataset (model implemented in PyTorch)

For more information on the model architecture, refer to:
    1. "Wide & Deep Learning for Recommender Systems" (https://arxiv.org/abs/1606.07792)
    2. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (https://arxiv.org/abs/2008.13535)

In order to fit a vanilla "Wide & Cross" model, set model_hyperParams_dict["n_cross_layers"]=0

This script fits both the "parallel" and the "stacked" variants of the "Deep & Cross" model

The data simulation class used to simulate the data is imported from the script "recsys_simulation.py"
Run "help(recsys_data_simulator)" in python in order to get more information on the data simulation code 
If you are not interested in the data simulation process, you can just treat the input data [model_data_df] as given, and proceed from the part of the script titled "## Deep & Cross Model - Implemented in PyTorch ##"

Intended future improvements of this script:
    1.  Make the model still work if certain feature sets are left out (e.g. no features to be embedded)
"""


import sys
import numpy as np
import pandas as pd
from tqdm import tqdm  # for a nice progress bar
from matplotlib import pyplot as plt
from sklearn.preprocessing import (
    # very good class for handling 1-hot feature encoding (especially for unseen categories in test data etc.)
    OneHotEncoder,
)
from pprint import pprint
import copy

import torch

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

sim_n_users = 5_000
sim_n_items = 50

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
        "max": 0.25,
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
    # this makes joining on these columns later a lot faster
    ["example_ID", "user_ID", "item_ID"],
    inplace=True,
)

# make some continuous features based on the context features and add them to the simulated data
# This simply adds redundant predictive information..
# ..but I include it in order to show the full functionality of the model
for context_x in tqdm(
    sim_obj.potential_context_attr, desc="creating some continuous features"
):
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

# add item attribute and user attribute data to data for model:
model_data_df = model_data_df.join(user_attr_df, on="user_ID").join(
    item_attr_df, on="item_ID"
)

model_data_df = model_data_df.reset_index().set_index(
    ["example_ID", "model_data_partition"]
)

## Deep & Cross Model - Implemented in PyTorch ## -----------------------------------------------
"""
-- Deep & Cross Model References --
1. "Wide & Deep Learning for Recommender Systems" (https://arxiv.org/abs/1606.07792)
2. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems" (https://arxiv.org/abs/2008.13535)
"""

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    pytorch_device = torch.device("mps")
else:
    pytorch_device = torch.device("cpu")
print(f"PyTorch is using device: {pytorch_device}")

# define model parameters:
# (in a real setting, these would likely be learned using hyper-parameter search or optimisation method)
model_hyperParams_dict = {
    "learning_rate": 0.001,
    "embed_dim": 10,  # all variables have the same embedding dimension (this could be trivially modified to have a different dimension for each variable)
    "deep_layer_structure": [
        # number of entries in this list is the number of hidden layers in the deep part of the model
        # tuple format is (n_nodes, activation_function)
        (100, torch.nn.ReLU()),
        (50, torch.nn.ReLU()),
        (25, torch.nn.ReLU()),
    ],
    "n_cross_layers": 0,  # 2,
    "train_batch_size": 5_000,
    "train_n_epochs": 100,
    "early_stopping_patience": 5,  # model training will stop if loss on validation data worsens for [early_stopping_patience] consecutive epochs
}

input_features_control_dict = {
    # this dictionary tells the model which features to include in the model, and exactly what to do with each
    # each feature will feed into one (or more) of the following parts of the model:
    #       1. "direct_to_deep":            (for continuous features): Feature fed straight into the deep part of the model, without any preprocessing
    #       2. "direct_to_cross":           (for continuous features): Feature fed straight into the cross part of the model, without any preprocessing
    #       3. "one_hot_then_deep":         (for categorical features): Feature 1-hot encoded then fed directly into the deep part of the model
    #       4. "embed_then_deep":           (for categorical features): Feature embedded then fed into the deep part of the model
    #       5. "one_hot_then_cross":        (for categorical features): Feature 1-hot encoded then fed into the cross (wide) part of the model
    #       6. "embed_then_cross":          (for categorical features): Feature embedded then fed into the cross (wide) part of the model
    "user_ID": {
        "category": "user_ID",
        "send_to": ["embed_then_deep", "embed_then_cross"],
    },
    "item_ID": {
        "category": "item_ID",
        "send_to": ["embed_then_deep", "embed_then_cross"],
    },
    "location": {
        "category": "user_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "age_group": {
        "category": "user_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "affluence": {
        "category": "user_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "main_device": {
        "category": "user_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "colour": {
        "category": "item_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "size": {
        "category": "item_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "material": {
        "category": "item_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "style": {
        "category": "item_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "time_of_day": {
        "category": "context_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross"],
    },
    "day_of_week": {
        "category": "context_attribute",
        "send_to": ["one_hot_then_deep", "one_hot_then_cross", "embed_then_deep"],
    },
    "social_context": {
        "category": "context_attribute",
        "send_to": ["one_hot_then_deep"],
    },
    "user_group_recommendation": {
        "category": "context_attribute",
        "send_to": ["one_hot_then_cross"],
    },
    "time_of_day_continuous": {
        "category": "context_attribute",
        "send_to": ["direct_to_deep", "direct_to_cross"],
    },
    "day_of_week_continuous": {
        "category": "context_attribute",
        "send_to": ["direct_to_deep", "direct_to_cross"],
    },
    "social_context_continuous": {
        "category": "context_attribute",
        "send_to": ["direct_to_deep", "direct_to_cross"],
    },
    "user_group_recommendation_continuous": {
        "category": "context_attribute",
        "send_to": ["direct_to_deep", "direct_to_cross"],
    },
}

# gather lists of feature names according to which part of the model they are input to:
X_name_list__direct_to_deep = [
    xName
    for xName in input_features_control_dict
    if "direct_to_deep" in input_features_control_dict[xName]["send_to"]
]
X_name_list__direct_to_cross = [
    xName
    for xName in input_features_control_dict
    if "direct_to_cross" in input_features_control_dict[xName]["send_to"]
]
X_name_list__one_hot_then_deep = [
    xName
    for xName in input_features_control_dict
    if "one_hot_then_deep" in input_features_control_dict[xName]["send_to"]
]
X_name_list__embed_then_deep = [
    xName
    for xName in input_features_control_dict
    if "embed_then_deep" in input_features_control_dict[xName]["send_to"]
]
X_name_list__one_hot_then_cross = [
    xName
    for xName in input_features_control_dict
    if "one_hot_then_cross" in input_features_control_dict[xName]["send_to"]
]
X_name_list__embed_then_cross = [
    xName
    for xName in input_features_control_dict
    if "embed_then_cross" in input_features_control_dict[xName]["send_to"]
]

# embeddings in PyTorch use an integer ID (e.g. n unique labels must use IDs 0,1,2,...,n for the embedding part of the model)
# this dictionary ("feature_embed_idx_ref_dict") is used to map each example from raw data ID (e.g. user_ID, item_ID etc.) to PyTorch embedding layer integer ID (and back again):
feature_embed_idx_ref_dict = {}
featureNames_to_embed = [
    x_name
    for x_name in set(X_name_list__embed_then_deep + X_name_list__embed_then_cross)
]
for x_name in featureNames_to_embed:
    feature_embed_idx_ref_dict[x_name] = {}
    unique_vals_list = (
        # all user_ID seen in training data
        model_data_df.loc[:, "train", :][x_name]
        .drop_duplicates()
        .values
    )
    np.random.shuffle(
        # wouldn't do this in a real application - just doing it here to make it clearer what the code is doing
        unique_vals_list
    )
    feature_embed_idx_ref_dict[x_name]["to_embed_ID"] = {
        unique_vals_list[i]: i for i in range(len(unique_vals_list))
    }
    feature_embed_idx_ref_dict[x_name]["from_embed_ID"] = {
        feature_embed_idx_ref_dict[x_name]["to_embed_ID"][val_id]: val_id
        for val_id in unique_vals_list
    }

# pprint(feature_embed_idx_ref_dict)

# one-hot encode categorical features:
one_hot_encoder_to_deep = OneHotEncoder(
    sparse_output=False,  # Will return sparse matrix if set True else will return an array
    handle_unknown="ignore",  # When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros
    dtype=np.float32,  # Desired data type of output
)
one_hot_encoder_to_cross = OneHotEncoder(
    sparse_output=False,  # Will return sparse matrix if set True else will return an array
    handle_unknown="ignore",  # When an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros
    dtype=np.float32,  # Desired data type of output
)
one_hot_encoder_to_deep.fit(
    model_data_df[X_name_list__one_hot_then_deep].loc[:, "train", :]
)
one_hot_encoder_to_cross.fit(
    model_data_df[X_name_list__one_hot_then_cross].loc[:, "train", :]
)
one_hot_to_deep_df = pd.DataFrame(
    one_hot_encoder_to_deep.transform(model_data_df[X_name_list__one_hot_then_deep]),
    columns=one_hot_encoder_to_deep.get_feature_names_out(),
    index=model_data_df.index,
)
one_hot_to_cross_df = pd.DataFrame(
    one_hot_encoder_to_cross.transform(model_data_df[X_name_list__one_hot_then_cross]),
    columns=one_hot_encoder_to_cross.get_feature_names_out(),
    index=model_data_df.index,
)

# convert embedding variables to their embedding IDs:
to_embed_IDs_df = model_data_df[feature_embed_idx_ref_dict.keys()].copy()
for embed_varname in feature_embed_idx_ref_dict:
    embed_ID_lookup_dict = feature_embed_idx_ref_dict[embed_varname]["to_embed_ID"]
    to_embed_IDs_df[embed_varname] = [
        embed_ID_lookup_dict[x] for x in to_embed_IDs_df[embed_varname]
    ]

# prepare data for model:
to_deep_df = one_hot_to_deep_df.join(
    model_data_df[X_name_list__direct_to_deep],
    on=["example_ID", "model_data_partition"],
)
to_cross_df = one_hot_to_cross_df.join(
    model_data_df[X_name_list__direct_to_cross],
    on=["example_ID", "model_data_partition"],
)
to_embed_then_deep_df = to_embed_IDs_df[X_name_list__embed_then_deep]
to_embed_then_cross_df = to_embed_IDs_df[X_name_list__embed_then_cross]

# put data into a PyTorch-friendly dataset class:
class dcn_model_pytorch_dataset_class(torch.utils.data.Dataset):
    """A torch dataset class is used to store data in a format that PyTorch is designed to interact with"""

    def __init__(
        self,
        y_vec,
        X_to_deep_df,
        X_to_cross_df,
        X_to_embed_then_deep_df,
        X_to_embed_then_cross_df,
    ):
        """
        TODO: proper documentation here
        """
        super(
            dcn_model_pytorch_dataset_class, self
        ).__init__()  # this allows us to inherit from the parent class (torch.utils.data.Dataset)
        self.y_vec = y_vec
        self.X_to_deep_df = X_to_deep_df
        self.X_to_cross_df = X_to_cross_df
        self.X_to_embed_then_deep_df = X_to_embed_then_deep_df
        self.X_to_embed_then_cross_df = X_to_embed_then_cross_df

    def __len__(self):
        """
        returns the total number of samples in the dataset
        """
        return len(self.y_vec)

    def __getitem__(self, idx):
        """
        loads and returns a single sample from the dataset at the given index [idx]
        """

        y_scalar = self.y_vec[idx]
        X_to_deep_row = self.X_to_deep_df.iloc[[idx]].values[0]
        X_to_cross_row = self.X_to_cross_df.iloc[[idx]].values[0]
        X_to_embed_then_deep_row = self.X_to_embed_then_deep_df.iloc[[idx]].values[0]
        X_to_embed_then_cross_row = self.X_to_embed_then_cross_df.iloc[[idx]].values[0]
        return (
            # response y (scalar):
            torch.tensor(y_scalar, dtype=torch.float32, device=pytorch_device),
            # features direct to the deep part of the model:
            torch.tensor(X_to_deep_row, dtype=torch.float32, device=pytorch_device),
            # features direct to the cross part of the model:
            torch.tensor(X_to_cross_row, dtype=torch.float32, device=pytorch_device),
            # features to be embedded then fed to the deep part of the model:
            torch.tensor(
                X_to_embed_then_deep_row, dtype=torch.int, device=pytorch_device
            ),
            # features to be embedded then fed to the cross part of the model:
            torch.tensor(
                X_to_embed_then_cross_row, dtype=torch.int, device=pytorch_device
            ),
        )


dcn_model_train_dataset = dcn_model_pytorch_dataset_class(
    y_vec=model_data_df.loc[:, "train", :]["bought"].values,
    X_to_deep_df=to_deep_df.loc[:, "train", :],
    X_to_cross_df=to_cross_df.loc[:, "train", :],
    X_to_embed_then_deep_df=to_embed_then_deep_df.loc[:, "train", :],
    X_to_embed_then_cross_df=to_embed_then_cross_df.loc[:, "train", :],
)

dcn_train_dataLoader = torch.utils.data.DataLoader(
    dataset=dcn_model_train_dataset,
    batch_size=model_hyperParams_dict["train_batch_size"],
    shuffle=True,
)


class DeepAndCross_Net_class(torch.nn.Module):
    def __init__(
        self,
        direct_to_deep_input_size,  # number of features in sample input features X to deep part of model (exluding variables to be embedded)
        direct_to_cross_input_size,  # number of features in sample input features X to cross part of model (excluding variables to be embedded)
        deep_layer_structure,  # refer to: model_hyperParams_dict["deep_layer_structure"]
        n_cross_layers,  # refer to model_hyperParams_dict["cross_layer_structure"]
        network_architecture,  # one of {"parallel","stacked"}
        embed_ID_lookup_dict,
        embed_then_deep_varnames_list,
        embed_then_cross_varnames_list,
        embed_dim,
    ):
        """
        TODO: proper documentation here

        embed_then_deep_varnames_list: list
            the original names of the variables are used for lookup in the embedding layer
            (the order of the variables must be the same as they appear in the training data)
        embed_then_cross_varnames_list: list
            the original names of the variables are used for lookup in the embedding layer
            (the order of the variables must be the same as they appear in the training data)
        """
        super().__init__()
        self.direct_to_deep_input_size = direct_to_deep_input_size
        self.direct_to_cross_input_size = direct_to_cross_input_size
        self.deep_layer_structure = deep_layer_structure
        self.n_cross_layers = n_cross_layers
        self.network_architecture = network_architecture
        self.embed_ID_lookup_dict = embed_ID_lookup_dict
        self.embed_then_deep_varnames_list = embed_then_deep_varnames_list
        self.embed_then_cross_varnames_list = embed_then_cross_varnames_list
        self.embed_dim = embed_dim

        assert self.network_architecture in [
            "parallel",
            "stacked",
        ], "network_architecture must be one of ['parallel','stacked']"

        self.embedding_layers_dict = torch.nn.ModuleDict()
        for embed_varname in self.embed_ID_lookup_dict.keys():
            self.embedding_layers_dict[embed_varname] = torch.nn.Embedding(
                num_embeddings=len(
                    self.embed_ID_lookup_dict[embed_varname]["to_embed_ID"]
                ),
                embedding_dim=self.embed_dim,
                device=pytorch_device,
            )

        self.to_cross_input_size = (
            self.direct_to_cross_input_size
            + len(self.embed_then_cross_varnames_list) * self.embed_dim
        )
        self.to_deep_input_size = (
            self.direct_to_deep_input_size
            + len(self.embed_then_deep_varnames_list) * self.embed_dim
        )
        if self.network_architecture == "stacked":
            # in the stacked architecture, the output of the cross network is included with the inputs to the deep network:
            self.to_deep_input_size = self.to_deep_input_size + self.to_cross_input_size

        self.deep_layers = torch.nn.ModuleList()
        current_to_deep_input_size = self.to_deep_input_size
        for n_nodes, activation_ftn in self.deep_layer_structure:
            self.deep_layers.append(
                torch.nn.Linear(current_to_deep_input_size, n_nodes)
            )
            current_to_deep_input_size = n_nodes  # size for next layer
            self.deep_layers.append(activation_ftn)

        self.cross_layers = torch.nn.ModuleList()
        for i in range(self.n_cross_layers):
            self.cross_layers.append(
                torch.nn.Linear(self.to_cross_input_size, self.to_cross_input_size)
            )

        if self.network_architecture == "stacked":
            # in the stacked architecture, the cross network output is included as input to the deep network
            self.linear_comb_model_outputs = torch.nn.Linear(
                in_features=self.deep_layer_structure[-1][0],
                out_features=1,
            )
        elif self.network_architecture == "parallel":
            # in the parallel architecture the outputs of the cross network and deep network are combined at the end
            self.linear_comb_model_outputs = torch.nn.Linear(
                in_features=self.deep_layer_structure[-1][0] + self.to_cross_input_size,
                out_features=1,
            )

    def forward(
        self, direct_to_deep_x, direct_to_cross_x, embed_then_deep_x, embed_then_cross_x
    ):
        """
        This function performs a forward pass through the network for a given batch of data

        Attributes
        ----------
        direct_to_deep_x: torch.Tensor (torch.float32)
            input features of deep part of model (exluding features to be embedded)
        direct_to_cross_x: torch.Tensor (torch.float32)
            input features to cross part of model (excluding features to be embedded)
        embed_then_deep_x: torch.Tensor (torch.int8)
            input IDs to look up in embedding table (for features to be embedded then fed to deep part of model)
        embed_then_cross_x: torch.Tensor (torch.int8)
            input IDs to look up in embedding table (for features to be embedded then fed to cross part of model)

        Returns
        -------
        model_output: torch.Tensor (torch.float32)
            vector of model predictions (each prediction in [0,1])
        """
        # look up embeddings for variables going to the deep part of the model:
        to_deep_embeddings = torch.stack(
            [
                self.embedding_layers_dict[self.embed_then_deep_varnames_list[i]](
                    embed_then_deep_x[:, i]
                )
                for i in range(len(self.embed_then_deep_varnames_list))
            ],
            axis=1,
        )
        # turn all embeddings into a single long vector within each sample
        to_deep_embeddings_flatten = torch.flatten(to_deep_embeddings, start_dim=1)

        # look up embeddings for variables going to the cross part of the model:
        to_cross_embeddings = torch.stack(
            [
                self.embedding_layers_dict[self.embed_then_cross_varnames_list[i]](
                    embed_then_cross_x[:, i]
                )
                for i in range(len(self.embed_then_cross_varnames_list))
            ],
            axis=1,
        )
        # turn all embeddings into a single long vector within each sample
        to_cross_embeddings_flatten = torch.flatten(to_cross_embeddings, start_dim=1)

        cross_x = torch.cat((to_cross_embeddings_flatten, direct_to_cross_x), axis=1)

        self.cross_x0 = cross_x
        for cross_layer in self.cross_layers:
            cross_x = self.cross_x0 * cross_layer(cross_x) + cross_x

        if self.network_architecture == "parallel":
            # in the parallel model, the cross and deep networks run in parallel
            deep_x = torch.cat((to_deep_embeddings_flatten, direct_to_deep_x), axis=1)
        elif self.network_architecture == "stacked":
            # in the stacked architecture, the output of the cross network is included in the input to the deep network
            deep_x = torch.cat(
                (to_deep_embeddings_flatten, direct_to_deep_x, cross_x), axis=1
            )

        # pass the input through the deep layers:
        for deep_layer in self.deep_layers:
            deep_x = deep_layer(deep_x)

        if self.network_architecture == "parallel":
            # in the parallel model, the cross and deep networks run in parallel, and we combine their outputs here:
            combine_outputs = torch.cat((deep_x, cross_x), axis=1)
        elif self.network_architecture == "stacked":
            # in the stacked architecture, the output of the cross network is included in the input to the deep network
            # so the combined network output is just the output of the deep network
            combine_outputs = deep_x

        linear_comb_model_outputs = self.linear_comb_model_outputs(combine_outputs)
        model_output = torch.flatten(torch.sigmoid(linear_comb_model_outputs))

        return model_output


# I initiate 1 model of each type:
# (1 with parallel architecture and 1 with stacked architecture)
parallel_DeepAndCross_Net = DeepAndCross_Net_class(
    direct_to_deep_input_size=dcn_model_train_dataset.__getitem__(idx=0)[1].shape[0],
    direct_to_cross_input_size=dcn_model_train_dataset.__getitem__(idx=0)[2].shape[0],
    deep_layer_structure=model_hyperParams_dict["deep_layer_structure"],
    n_cross_layers=model_hyperParams_dict["n_cross_layers"],
    network_architecture="parallel",
    embed_ID_lookup_dict=feature_embed_idx_ref_dict,
    embed_then_deep_varnames_list=list(to_embed_then_deep_df.columns),
    embed_then_cross_varnames_list=list(to_embed_then_cross_df.columns),
    embed_dim=model_hyperParams_dict["embed_dim"],
)

stacked_DeepAndCross_Net = DeepAndCross_Net_class(
    direct_to_deep_input_size=dcn_model_train_dataset.__getitem__(idx=0)[1].shape[0],
    direct_to_cross_input_size=dcn_model_train_dataset.__getitem__(idx=0)[2].shape[0],
    deep_layer_structure=model_hyperParams_dict["deep_layer_structure"],
    n_cross_layers=model_hyperParams_dict["n_cross_layers"],
    network_architecture="stacked",
    embed_ID_lookup_dict=feature_embed_idx_ref_dict,
    embed_then_deep_varnames_list=list(to_embed_then_deep_df.columns),
    embed_then_cross_varnames_list=list(to_embed_then_cross_df.columns),
    embed_dim=model_hyperParams_dict["embed_dim"],
)

# push models to GPU (if available):
parallel_DeepAndCross_Net.to(
    pytorch_device
)  # enables model training on the GPU (if it is available)
stacked_DeepAndCross_Net.to(
    pytorch_device
)  # enables model training on the GPU (if it is available)

# store full training dataset:
# (for model evaluation after completing each epoch)
torch_X_full_trainData = {
    "y": torch.tensor(
        model_data_df.loc[:, "train", :]["bought"].values,
        dtype=torch.float32,
        device=pytorch_device,
    ),
    "to_deep": torch.tensor(
        to_deep_df.loc[:, "train", :].to_numpy(),
        dtype=torch.float32,
        device=pytorch_device,
    ),
    "to_cross": torch.tensor(
        to_cross_df.loc[:, "train", :].to_numpy(),
        dtype=torch.float32,
        device=pytorch_device,
    ),
    "to_embed_then_deep": torch.tensor(
        to_embed_then_deep_df.loc[:, "train", :].to_numpy(),
        dtype=torch.int,
        device=pytorch_device,
    ),
    "to_embed_then_cross": torch.tensor(
        to_embed_then_cross_df.loc[:, "train", :].to_numpy(),
        dtype=torch.int,
        device=pytorch_device,
    ),
}
# store full validation dataset:
# (for model evaluation after completing each epoch)
torch_X_full_validData = {
    "y": torch.tensor(
        model_data_df.loc[:, "validate", :]["bought"].values,
        dtype=torch.float32,
        device=pytorch_device,
    ),
    "to_deep": torch.tensor(
        to_deep_df.loc[:, "validate", :].to_numpy(),
        dtype=torch.float32,
        device=pytorch_device,
    ),
    "to_cross": torch.tensor(
        to_cross_df.loc[:, "validate", :].to_numpy(),
        dtype=torch.float32,
        device=pytorch_device,
    ),
    "to_embed_then_deep": torch.tensor(
        to_embed_then_deep_df.loc[:, "validate", :].to_numpy(),
        dtype=torch.int,
        device=pytorch_device,
    ),
    "to_embed_then_cross": torch.tensor(
        to_embed_then_cross_df.loc[:, "validate", :].to_numpy(),
        dtype=torch.int,
        device=pytorch_device,
    ),
}

# train the parallel architecture --------------------------------
parallel_dcn__loss_func = torch.nn.BCELoss()
parallel_dcn__optimizer = torch.optim.Adam(
    parallel_DeepAndCross_Net.parameters(),
    lr=model_hyperParams_dict["learning_rate"],
)

parallel_dcn__train_loss_history = []
parallel_dcn__valid_loss_history = []

parallel_dcn__early_stopping_patience_counter = 0
parallel_dcn__best_valid_loss_so_far = None
parallel_dcn__best_model_params_so_far = None

for epoch in range(1, model_hyperParams_dict["train_n_epochs"] + 1):
    for batch_num, batch_data in enumerate(dcn_train_dataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        (
            true_y,
            X_direct_to_deep,
            X_direct_to_cross,
            X_embed_then_deep,
            X_embed_then_cross,
        ) = batch_data

        # zero the parameter gradients
        parallel_dcn__optimizer.zero_grad()

        # forward + backward + optimize
        parallel_dcn_preds = parallel_DeepAndCross_Net(
            direct_to_deep_x=X_direct_to_deep,
            direct_to_cross_x=X_direct_to_cross,
            embed_then_deep_x=X_embed_then_deep,
            embed_then_cross_x=X_embed_then_cross,
        )
        parallel_dcn__loss_val = parallel_dcn__loss_func(parallel_dcn_preds, true_y)
        parallel_dcn__loss_val.backward()
        parallel_dcn__optimizer.step()

    # at end of each epoch:
    # calculate loss over full training set:
    parallel_dcn__model_preds_full_trainData = parallel_DeepAndCross_Net(
        direct_to_deep_x=torch_X_full_trainData["to_deep"],
        direct_to_cross_x=torch_X_full_trainData["to_cross"],
        embed_then_deep_x=torch_X_full_trainData["to_embed_then_deep"],
        embed_then_cross_x=torch_X_full_trainData["to_embed_then_cross"],
    )
    parallel_dcn__train_loss = parallel_dcn__loss_func(
        parallel_dcn__model_preds_full_trainData,
        torch_X_full_trainData["y"],
    ).item()
    parallel_dcn__train_loss_history.append(parallel_dcn__train_loss)

    # calculate loss over full validation set:
    parallel_dcn__model_preds_full_validData = parallel_DeepAndCross_Net(
        direct_to_deep_x=torch_X_full_validData["to_deep"],
        direct_to_cross_x=torch_X_full_validData["to_cross"],
        embed_then_deep_x=torch_X_full_validData["to_embed_then_deep"],
        embed_then_cross_x=torch_X_full_validData["to_embed_then_cross"],
    )
    parallel_dcn__valid_loss = parallel_dcn__loss_func(
        parallel_dcn__model_preds_full_validData,
        torch_X_full_validData["y"],
    ).item()
    parallel_dcn__valid_loss_history.append(parallel_dcn__valid_loss)

    # check for consecutive degradation in validation loss (early stopping):
    if len(parallel_dcn__valid_loss_history) > 1 and (
        parallel_dcn__valid_loss > parallel_dcn__valid_loss_history[-2]
    ):
        parallel_dcn__early_stopping_patience_counter += 1
    else:
        parallel_dcn__early_stopping_patience_counter = 0

    # store parameters of best model found so far:
    if (
        parallel_dcn__best_valid_loss_so_far is None
        or parallel_dcn__valid_loss < parallel_dcn__best_valid_loss_so_far
    ):
        parallel_dcn__best_valid_loss_so_far = parallel_dcn__valid_loss
        parallel_dcn__best_model_params_so_far = copy.deepcopy(
            parallel_DeepAndCross_Net.state_dict()
        )

    print(
        f"""
        -- finished epoch {epoch} of {model_hyperParams_dict['train_n_epochs']}-- 
        loss on training data:      {parallel_dcn__train_loss:.3f}
        loss on validation data:    {parallel_dcn__valid_loss:.3f}
        """
    )
    if (
        parallel_dcn__early_stopping_patience_counter
        >= model_hyperParams_dict["early_stopping_patience"]
    ):
        print(
            f"""
        validation loss has increased in each of the last {model_hyperParams_dict['early_stopping_patience']} epochs
        -- model training stopped --
        """
        )
        break

print("-- Finished Parallel Model Training --\n")

# load parameters of best model found (lowest validation loss):
print(
    """Deep & Cross Model: Parallel architecture:
    restoring parameters of best (lowest valid loss) model found during training...""",
    end="",
)
parallel_DeepAndCross_Net.load_state_dict(parallel_dcn__best_model_params_so_far)
print("done")


# train the stacked architecture ---------------------------------
stacked_dcn__loss_func = torch.nn.BCELoss()
stacked_dcn__optimizer = torch.optim.Adam(
    stacked_DeepAndCross_Net.parameters(),
    lr=model_hyperParams_dict["learning_rate"],
)

stacked_dcn__train_loss_history = []
stacked_dcn__valid_loss_history = []

stacked_dcn__early_stopping_patience_counter = 0
stacked_dcn__best_valid_loss_so_far = None
stacked_dcn__best_model_params_so_far = None

for epoch in range(1, model_hyperParams_dict["train_n_epochs"] + 1):
    for batch_num, batch_data in enumerate(dcn_train_dataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        (
            true_y,
            X_direct_to_deep,
            X_direct_to_cross,
            X_embed_then_deep,
            X_embed_then_cross,
        ) = batch_data

        # zero the parameter gradients
        stacked_dcn__optimizer.zero_grad()

        # forward + backward + optimize
        stacked_dcn_preds = stacked_DeepAndCross_Net(
            direct_to_deep_x=X_direct_to_deep,
            direct_to_cross_x=X_direct_to_cross,
            embed_then_deep_x=X_embed_then_deep,
            embed_then_cross_x=X_embed_then_cross,
        )
        stacked_dcn__loss_val = stacked_dcn__loss_func(stacked_dcn_preds, true_y)
        stacked_dcn__loss_val.backward()
        stacked_dcn__optimizer.step()

    # at end of each epoch:
    # calculate loss over full training set:
    stacked_dcn__model_preds_full_trainData = stacked_DeepAndCross_Net(
        direct_to_deep_x=torch_X_full_trainData["to_deep"],
        direct_to_cross_x=torch_X_full_trainData["to_cross"],
        embed_then_deep_x=torch_X_full_trainData["to_embed_then_deep"],
        embed_then_cross_x=torch_X_full_trainData["to_embed_then_cross"],
    )
    stacked_dcn__train_loss = stacked_dcn__loss_func(
        stacked_dcn__model_preds_full_trainData,
        torch_X_full_trainData["y"],
    ).item()
    stacked_dcn__train_loss_history.append(stacked_dcn__train_loss)

    # calculate loss over full validation set:
    stacked_dcn__model_preds_full_validData = stacked_DeepAndCross_Net(
        direct_to_deep_x=torch_X_full_validData["to_deep"],
        direct_to_cross_x=torch_X_full_validData["to_cross"],
        embed_then_deep_x=torch_X_full_validData["to_embed_then_deep"],
        embed_then_cross_x=torch_X_full_validData["to_embed_then_cross"],
    )
    stacked_dcn__valid_loss = stacked_dcn__loss_func(
        stacked_dcn__model_preds_full_validData,
        torch_X_full_validData["y"],
    ).item()
    stacked_dcn__valid_loss_history.append(stacked_dcn__valid_loss)

    # check for consecutive degradation in validation loss (early stopping):
    if len(stacked_dcn__valid_loss_history) > 1 and (
        stacked_dcn__valid_loss > stacked_dcn__valid_loss_history[-2]
    ):
        stacked_dcn__early_stopping_patience_counter += 1
    else:
        stacked_dcn__early_stopping_patience_counter = 0

    # store parameters of best model found so far:
    if (
        stacked_dcn__best_valid_loss_so_far is None
        or stacked_dcn__valid_loss < stacked_dcn__best_valid_loss_so_far
    ):
        stacked_dcn__best_valid_loss_so_far = stacked_dcn__valid_loss
        stacked_dcn__best_model_params_so_far = copy.deepcopy(
            stacked_DeepAndCross_Net.state_dict()
        )

    print(
        f"""
        -- finished epoch {epoch} of {model_hyperParams_dict['train_n_epochs']}-- 
        loss on training data:      {stacked_dcn__train_loss:.3f}
        loss on validation data:    {stacked_dcn__valid_loss:.3f}
        """
    )
    if (
        stacked_dcn__early_stopping_patience_counter
        >= model_hyperParams_dict["early_stopping_patience"]
    ):
        print(
            f"""
        validation loss has increased in each of the last {model_hyperParams_dict['early_stopping_patience']} epochs
        -- model training stopped --
        """
        )
        break

print("-- Finished Stacked Model Training --\n")

# load parameters of best model found (lowest validation loss):
print(
    """Deep & Cross Model: Stacked architecture:
    restoring parameters of best (lowest valid loss) model found during training...""",
    end="",
)
stacked_DeepAndCross_Net.load_state_dict(stacked_dcn__best_model_params_so_far)
print("done")

# plot loss during model training (of both models):
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
axs[0].plot(
    range(1, len(parallel_dcn__train_loss_history) + 1),
    parallel_dcn__train_loss_history,
    label="training (parallel architecture)",
)
axs[0].plot(
    range(1, len(parallel_dcn__valid_loss_history) + 1),
    parallel_dcn__valid_loss_history,
    label="validation (parallel architecture)",
)
axs[0].plot(
    range(1, len(stacked_dcn__train_loss_history) + 1),
    stacked_dcn__train_loss_history,
    label="training (stacked architecture)",
)
axs[0].plot(
    range(1, len(stacked_dcn__valid_loss_history) + 1),
    stacked_dcn__train_loss_history,
    label="validation (stacked architecture)",
)
axs[0].set_title("Model Training: Both Models")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss (Binary Cross Entropy)")
axs[0].legend()
axs[1].plot(
    range(1, len(parallel_dcn__train_loss_history) + 1),
    parallel_dcn__train_loss_history,
    label="training",
)
axs[1].plot(
    range(1, len(parallel_dcn__valid_loss_history) + 1),
    parallel_dcn__valid_loss_history,
    label="validation",
)
axs[1].set_title("Model Training: Parallel Architecture")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss (Binary Cross Entropy)")
axs[1].legend()
axs[2].plot(
    range(1, len(stacked_dcn__train_loss_history) + 1),
    stacked_dcn__train_loss_history,
    label="training",
)
axs[2].plot(
    range(1, len(stacked_dcn__valid_loss_history) + 1),
    stacked_dcn__valid_loss_history,
    label="validation",
)
axs[2].set_title("Model Training: Stacked Architecture")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Loss (Binary Cross Entropy)")
axs[2].legend()

# generate recommendations for a random user:
# (can only recommend items not already bought by that user)
"""
note that this is not how we would generate recommendations for users in a real application
the code which follows here is highly suboptimal - it just serves here to illustrate heuristically that the models are working
in a real application, data would be batch-processed into the required format for the model - in order to achieve minimal latency in generating recommendations
"""
random_user_ID = model_data_df.sample(1)["user_ID"].item()
random_context = sim_obj.generate_random_context()
all_item_IDs_set = set(item_attr_df.index)
item_IDs_actually_bought_set = set(
    model_data_df.query(f"user_ID=={random_user_ID} & bought==1")
    .item_ID.drop_duplicates()
    .values
)
recommendable_item_IDs_set = all_item_IDs_set.difference(item_IDs_actually_bought_set)
pred_features_df = (
    pd.DataFrame(
        {
            "user_ID": random_user_ID,
            "item_ID": list(recommendable_item_IDs_set),
        }
    )
    .set_index(["user_ID", "item_ID"])
    .join(item_attr_df, on="item_ID")
    .join(user_attr_df, on="user_ID")
)
for context_categ in random_context:
    categ_val = random_context[context_categ]
    pred_features_df[context_categ] = categ_val
    # also need to add those simulated continuous features
    # (I just added these features in order to illustrate the full functionality of the model)
    # DEFINITELY wouldn't do this in a real application of the model
    conts_categ_vals = model_data_df.query(f"{context_categ}=='{categ_val}'")[
        f"{context_categ}_continuous"
    ].values
    pred_features_df[f"{context_categ}_continuous"] = np.random.uniform(
        low=min(conts_categ_vals),
        high=max(conts_categ_vals),
        size=len(pred_features_df),
    )
pred_features_df["true_y"] = [
    sim_obj.calc_user_preference_for_item(
        user_id=random_user_ID,
        item_id=ix,
        recommend_context=random_context,
    )["rating_in_this_context"]
    for ix in pred_features_df.reset_index()["item_ID"]
]
pred_features_df["true_y_rank"] = (
    pred_features_df["true_y"].rank(ascending=False).astype(int)
)

# get data into format expected by pytorch model:
one_hot_to_deep_df = pd.DataFrame(
    one_hot_encoder_to_deep.transform(pred_features_df[X_name_list__one_hot_then_deep]),
    columns=one_hot_encoder_to_deep.get_feature_names_out(),
    index=pred_features_df.index,
)
one_hot_to_cross_df = pd.DataFrame(
    one_hot_encoder_to_cross.transform(
        pred_features_df[X_name_list__one_hot_then_cross]
    ),
    columns=one_hot_encoder_to_cross.get_feature_names_out(),
    index=pred_features_df.index,
)
to_embed_IDs_df = pred_features_df.reset_index()[
    feature_embed_idx_ref_dict.keys()
].copy()
for embed_varname in feature_embed_idx_ref_dict:
    embed_ID_lookup_dict = feature_embed_idx_ref_dict[embed_varname]["to_embed_ID"]
    to_embed_IDs_df[embed_varname] = [
        embed_ID_lookup_dict[x] for x in to_embed_IDs_df[embed_varname]
    ]

to_deep_df = one_hot_to_deep_df.join(
    pred_features_df[X_name_list__direct_to_deep],
    on=["user_ID", "item_ID"],
)
to_cross_df = one_hot_to_cross_df.join(
    pred_features_df[X_name_list__direct_to_cross],
    on=["user_ID", "item_ID"],
)
to_embed_then_deep_df = to_embed_IDs_df[X_name_list__embed_then_deep]
to_embed_then_cross_df = to_embed_IDs_df[X_name_list__embed_then_cross]

pred_features_df["pred_y_parallel"] = (
    parallel_DeepAndCross_Net(
        direct_to_deep_x=torch.tensor(
            to_deep_df.to_numpy(), dtype=torch.float32, device=pytorch_device
        ),
        direct_to_cross_x=torch.tensor(
            to_cross_df.to_numpy(), dtype=torch.float32, device=pytorch_device
        ),
        embed_then_deep_x=torch.tensor(
            to_embed_then_deep_df.to_numpy(), dtype=torch.int, device=pytorch_device
        ),
        embed_then_cross_x=torch.tensor(
            to_embed_then_cross_df.to_numpy(), dtype=torch.int, device=pytorch_device
        ),
    )
    .cpu()
    .detach()
    .numpy()
)
pred_features_df["pred_y_parallel_rank"] = (
    pred_features_df["pred_y_parallel"].rank(ascending=False).astype(int)
)
pred_features_df["pred_y_stacked"] = (
    stacked_DeepAndCross_Net(
        direct_to_deep_x=torch.tensor(
            to_deep_df.to_numpy(), dtype=torch.float32, device=pytorch_device
        ),
        direct_to_cross_x=torch.tensor(
            to_cross_df.to_numpy(), dtype=torch.float32, device=pytorch_device
        ),
        embed_then_deep_x=torch.tensor(
            to_embed_then_deep_df.to_numpy(), dtype=torch.int, device=pytorch_device
        ),
        embed_then_cross_x=torch.tensor(
            to_embed_then_cross_df.to_numpy(), dtype=torch.int, device=pytorch_device
        ),
    )
    .cpu()
    .detach()
    .numpy()
)
pred_features_df["pred_y_stacked_rank"] = (
    pred_features_df["pred_y_stacked"].rank(ascending=False).astype(int)
)

fig, axs = plt.subplots(figsize=(11, 5), nrows=1, ncols=2)
axs[0].scatter(
    pred_features_df["pred_y_parallel_rank"],
    pred_features_df["true_y_rank"],
    c=pred_features_df["true_y_rank"],
)
axs[0].set_xlabel("predicted_rank")
axs[0].set_ylabel("true_rank")
axs[0].set_title("Parallel Architecture")
axs[1].scatter(
    pred_features_df["pred_y_stacked_rank"],
    pred_features_df["true_y_rank"],
    c=pred_features_df["true_y_rank"],
)
axs[1].set_xlabel("predicted_rank")
axs[1].set_ylabel("true_rank")
axs[1].set_title("Stacked Architecture")
open_curly_char = "{"
close_curly_char = "}"
fig.suptitle(
    f"predicted vs. actual item affinity for user_ID={random_user_ID} in context {open_curly_char}{','.join(random_context.values())}{close_curly_char} (removed items already bought by user)"
)
