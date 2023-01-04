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

sim_n_users = 100
sim_n_items = 10

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
    "embed_dim": "TODO",  # should this be different per variable?
    "deep_layer_structure": [
        # number of entries in this list is the number of hidden layers in the deep part of the model
        # tuple format is (n_nodes, activation_function)
        (100, torch.nn.ReLU()),
        (50, torch.nn.ReLU()),
        (25, torch.nn.ReLU()),
    ],
    "n_cross_layers": 2,
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

# put data into a PyTorch-friendly dataset class:
class dcn_model_pytorch_dataset_class(torch.utils.data.Dataset):
    """A torch dataset class is used to store data in a format that PyTorch is designed to interact with"""

    def __init__(
        self,
        y_vec,
        X_to_deep_df,
        X_to_cross_df,
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

    def __len__(self):
        """
        returns the number of samples in our dataset
        """
        return len(self.y_vec)

    def __getitem__(self, idx):
        """
        loads and returns a single sample from the dataset at the given index [idx]
        """

        X_to_deep_row = self.X_to_deep_df.iloc[[idx]].values[0]
        X_to_cross_row = self.X_to_cross_df.iloc[[idx]].values[0]
        y_scalar = self.y_vec[idx]

        return (
            torch.tensor(y_scalar, dtype=torch.float32, device=pytorch_device),
            torch.tensor(X_to_deep_row, dtype=torch.float32, device=pytorch_device),
            torch.tensor(X_to_cross_row, dtype=torch.float32, device=pytorch_device),
            # [], # embed_then_deep
            # [], # embed_then_cross
        )


dcn_model_train_dataset = dcn_model_pytorch_dataset_class(
    y_vec=model_data_df.loc[:, "train", :]["bought"].values,
    X_to_deep_df=to_deep_df.loc[:, "train", :],
    X_to_cross_df=to_cross_df.loc[:, "train", :],
)

dcn_train_dataLoader = torch.utils.data.DataLoader(
    dataset=dcn_model_train_dataset,
    batch_size=model_hyperParams_dict["train_batch_size"],
    shuffle=True,
)


class DeepAndCross_Net_class(torch.nn.Module):
    def __init__(
        self,
        to_deep_input_size,  # number of features in sample input features X to deep part of model (exluding embeddings)
        to_cross_input_size,  # number of features in sample input features X to cross part of model (excluding embeddings)
        deep_layer_structure,  # refer to: model_hyperParams_dict["deep_layer_structure"]
        n_cross_layers,  # refer to model_hyperParams_dict["cross_layer_structure"]
        network_architecture,  # one of {"parallel","stacked"}
    ):
        """
        TODO: proper documentation here
        """
        super().__init__()
        self.to_deep_input_size = to_deep_input_size
        self.to_cross_input_size = to_cross_input_size
        self.deep_layer_structure = deep_layer_structure
        self.n_cross_layers = n_cross_layers
        self.network_architecture = network_architecture
        self.x0 = None

        if self.network_architecture == "stacked":
            print(
                f"network architecture '{self.network_architecture}' not implemented yet"
            )
            self.network_architecture = "parallel"
            print(f"=> using network architecture '{self.network_architecture}'")

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

        self.linear_comb_model_outputs = torch.nn.Linear(
            in_features=self.deep_layer_structure[-1][0] + self.to_cross_input_size,
            out_features=1,
        )

    def forward(self, deep_x, cross_x):
        """
        TODO: proper documentation here
        """
        self.cross_x0 = cross_x
        for deep_layer in self.deep_layers:
            deep_x = deep_layer(deep_x)
        for cross_layer in self.cross_layers:
            cross_x = self.cross_x0 * cross_layer(cross_x) + cross_x

        combine_outputs = torch.cat((deep_x, cross_x), axis=1)
        linear_comb_model_outputs = self.linear_comb_model_outputs(combine_outputs)
        model_output = torch.flatten(torch.sigmoid(linear_comb_model_outputs))

        return model_output


DeepAndCross_Net = DeepAndCross_Net_class(
    to_deep_input_size=dcn_model_train_dataset.__getitem__(idx=0)[1].shape[0],
    to_cross_input_size=dcn_model_train_dataset.__getitem__(idx=0)[2].shape[0],
    deep_layer_structure=model_hyperParams_dict["deep_layer_structure"],
    n_cross_layers=model_hyperParams_dict["n_cross_layers"],
    network_architecture="parallel",
)

DeepAndCross_Net.to(
    pytorch_device
)  # enables model training on the GPU (if it is available)

loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(
    DeepAndCross_Net.parameters(),
    lr=model_hyperParams_dict["learning_rate"],
)

train_loss_history = []  # store loss on training data in each epoch
valid_loss_history = []  # store loss on validation data in each epoch

early_stopping_patience_counter = 0
best_valid_loss_so_far = None
best_model_params_so_far = None

for epoch in range(1, model_hyperParams_dict["train_n_epochs"] + 1):
    for batch_num, batch_data in enumerate(dcn_train_dataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        true_y, X_to_deep, X_to_cross = batch_data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        model_preds = DeepAndCross_Net(deep_x=X_to_deep, cross_x=X_to_cross)
        loss_val = loss_func(model_preds, true_y)
        loss_val.backward()
        optimizer.step()

    # calculate loss over full training set:
    model_preds_full_trainData = DeepAndCross_Net(
        deep_x=torch.tensor(
            to_deep_df.loc[:, "train", :].to_numpy(),
            dtype=torch.float32,
            device=pytorch_device,
        ),
        cross_x=torch.tensor(
            to_cross_df.loc[:, "train", :].to_numpy(),
            dtype=torch.float32,
            device=pytorch_device,
        ),
    )
    train_loss = loss_func(
        model_preds_full_trainData,
        torch.tensor(
            model_data_df.loc[:, "train", :]["bought"].values,
            dtype=torch.float32,
            device=pytorch_device,
        ),
    ).item()
    train_loss_history.append(train_loss)

    # calculate loss over full validation set:
    model_preds_full_validData = DeepAndCross_Net(
        deep_x=torch.tensor(
            to_deep_df.loc[:, "validate", :].to_numpy(),
            dtype=torch.float32,
            device=pytorch_device,
        ),
        cross_x=torch.tensor(
            to_cross_df.loc[:, "validate", :].to_numpy(),
            dtype=torch.float32,
            device=pytorch_device,
        ),
    )
    valid_loss = loss_func(
        model_preds_full_validData,
        torch.tensor(
            model_data_df.loc[:, "validate", :]["bought"].values,
            dtype=torch.float32,
            device=pytorch_device,
        ),
    ).item()
    valid_loss_history.append(valid_loss)

    # check for consecutive degradation in validation loss (early stopping):
    if len(valid_loss_history) > 1 and (valid_loss > valid_loss_history[-2]):
        early_stopping_patience_counter += 1
    else:
        early_stopping_patience_counter = 0

    # store parameters of best model found so far:
    if best_valid_loss_so_far is None or valid_loss < best_valid_loss_so_far:
        best_valid_loss_so_far = valid_loss
        best_model_params_so_far = copy.deepcopy(DeepAndCross_Net.state_dict())

    print(
        f"""
        -- finished epoch {epoch} of {model_hyperParams_dict['train_n_epochs']}-- 
        loss on training data:      {train_loss:.3f}
        loss on validation data:    {valid_loss:.3f}
        """
    )
    if (
        early_stopping_patience_counter
        >= model_hyperParams_dict["early_stopping_patience"]
    ):
        print(
            f"""
        validation loss has increased in each of the last {model_hyperParams_dict['early_stopping_patience']} epochs
        -- model training stopped --
        """
        )
        break

print("-- Finished Model Training --")

# load parameters of best model found (lowest validation loss):
DeepAndCross_Net.load_state_dict(best_model_params_so_far)

# plot loss during model training:
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(train_loss_history) + 1), train_loss_history, label="training data"
)
plt.plot(
    range(1, len(valid_loss_history) + 1), valid_loss_history, label="validation data"
)
plt.xlabel("Epoch")
plt.ylabel("Loss (Binary Cross Entropy)")
plt.title("Per-Epoch Loss During Model Training (Binary Cross Entropy Loss)")
plt.legend()
