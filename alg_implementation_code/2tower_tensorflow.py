import sys
import numpy as np
import pandas as pd
from tqdm import tqdm  # for a nice progress bar
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import (
    # very good class for handling 1-hot feature encoding (especially for unseen categories in test data etc.)
    OneHotEncoder,
)

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

sim_n_users = 10_000
sim_n_items = 200

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

## 2-Tower Model - Implemented in TensorFlow ## -----------------------------------------------
# (actually I've defined a 3-tower model, but you can easily remove the "recommendation context" tower to get the standard 2-tower model)

# define which features will be used by the model:
# (and which features are user-based, item-based, context-based)
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

# define model parameters:
# (in a real setting, these would likely be learned using hyper-parameter search or optimisation method)
model_hyperParams_dict = {
    "learning_rate": 0.0001,
    "embed_dim": 5,  # dimension of latent user and item embeddings
    "encoder_output_dim": 100,  # dimension of output of each tower
    "tower_depth": 3,  # number of hidden layers in each tower
    "tower_width": 100,  # number of nodes in each hidden layer (in each tower)
    "n_train_epochs": 200,
    "dropout_percent": 0.01,  # Value in (0.0,1.0). Dropout to apply between each layer in each tower
    "train_batch_size": 10_000,
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
# NOTE: if a category only appears in the validation set, then this could cause data leakage
# (I am ignoring this issue here for the sake of simplicity)
sk_1hotEncoder_user_attr.fit(user_attr_df)
sk_1hotEncoder_item_attr.fit(item_attr_df)
sk_1hotEncoder_context_attr.fit(
    model_data_df.loc[:, :, :, "train"][
        include_featureNames_dict["context_feature_names"]
    ].drop_duplicates()
)

# create the input data for the model:
X_user_IDs = pd.DataFrame(
    # I use this method in order to preserve the multi-index for later reference
    model_data_df.index.get_level_values("user_ID"),
    index=model_data_df.index,
)
X_item_IDs = pd.DataFrame(
    # I use this method in order to preserve the multi-index for later reference
    model_data_df.index.get_level_values("item_ID"),
    index=model_data_df.index,
)
X_user_attr_raw = model_data_df.join(user_attr_df, on="user_ID")[
    include_featureNames_dict["user_feature_names"]
]
X_user_attr_1hot = pd.DataFrame(
    sk_1hotEncoder_user_attr.transform(X_user_attr_raw),
    columns=sk_1hotEncoder_user_attr.get_feature_names_out(),
    index=X_user_attr_raw.index,
)
X_item_attr_raw = model_data_df.join(item_attr_df, on="item_ID")[
    include_featureNames_dict["item_feature_names"]
]
X_item_attr_1hot = pd.DataFrame(
    sk_1hotEncoder_item_attr.transform(X_item_attr_raw),
    columns=sk_1hotEncoder_item_attr.get_feature_names_out(),
    index=X_item_attr_raw.index,
)
X_context_attr_raw = model_data_df[include_featureNames_dict["context_feature_names"]]
X_context_attr_1hot = pd.DataFrame(
    sk_1hotEncoder_context_attr.transform(X_context_attr_raw),
    columns=sk_1hotEncoder_context_attr.get_feature_names_out(),
    index=X_context_attr_raw.index,
)

## define the 2-Tower TensorFlow.Keras model ##
tf_user_embedding_lookup = tf.keras.layers.Embedding(
    input_dim=sim_n_users,  # number of unique user_ID in our user database
    output_dim=model_hyperParams_dict["embed_dim"],  # dimension of embedding vector
    embeddings_initializer="he_normal",
    name="user_embedding_lookup",
    input_length=1,
)
tf_item_embedding_lookup = tf.keras.layers.Embedding(
    input_dim=sim_n_items,  # number of unique item_ID in our item catalogue
    output_dim=model_hyperParams_dict["embed_dim"],  # dimension of embedding vector
    embeddings_initializer="he_normal",
    name="item_embedding_lookup",
)

# create user tower:
tf_user_ID_input = keras.layers.Input(shape=(1,), name="user_ID_input")
tf_user_feature_input = keras.layers.Input(
    shape=(X_user_attr_1hot.shape[1],), name="user_feature_input"
)
tf_user_embedding = tf_user_embedding_lookup(tf_user_ID_input)
tf_user_embedding_flatten = keras.layers.Flatten(name="flatten_user_embedding")(
    tf_user_embedding
)
tf_user_x = keras.layers.Concatenate(axis=1, name="concat_user_tower_input")(
    [tf_user_embedding_flatten, tf_user_feature_input],
)
for layer_i in range(1, model_hyperParams_dict["tower_depth"] + 1):
    tf_user_x = keras.layers.Dense(
        units=model_hyperParams_dict["tower_width"],
        activation="relu",
        name=f"user_dense_{layer_i}",
    )(tf_user_x)
    tf_user_x = keras.layers.Dropout(
        model_hyperParams_dict["dropout_percent"], name=f"user_dropOut_{layer_i}"
    )(tf_user_x)
tf_user_tower_output = keras.layers.Dense(
    units=model_hyperParams_dict["encoder_output_dim"],
    activation="relu",
    name=f"user_tower_output",
)(tf_user_x)

# create item tower:
tf_item_ID_input = keras.layers.Input(shape=(1,), name="item_ID_input")
tf_item_feature_input = keras.layers.Input(
    shape=(X_item_attr_1hot.shape[1],), name="item_feature_input"
)
tf_item_embedding = tf_item_embedding_lookup(tf_item_ID_input)
tf_item_embedding_flatten = keras.layers.Flatten(name="flatten_item_embedding")(
    tf_item_embedding
)
tf_item_x = keras.layers.Concatenate(axis=1, name="concat_item_tower_input")(
    [tf_item_embedding_flatten, tf_item_feature_input],
)
for layer_i in range(1, model_hyperParams_dict["tower_depth"] + 1):
    tf_item_x = keras.layers.Dense(
        units=model_hyperParams_dict["tower_width"],
        activation="relu",
        name=f"item_dense_{layer_i}",
    )(tf_item_x)
    tf_item_x = keras.layers.Dropout(
        model_hyperParams_dict["dropout_percent"], name=f"item_dropOut_{layer_i}"
    )(tf_item_x)
tf_item_tower_output = keras.layers.Dense(
    units=model_hyperParams_dict["encoder_output_dim"],
    activation="relu",
    name=f"item_tower_output",
)(tf_item_x)

# create context tower:
tf_context_feature_input = keras.layers.Input(
    shape=(X_context_attr_1hot.shape[1],), name="context_feature_input"
)
tf_context_x = tf_context_feature_input
for layer_i in range(1, model_hyperParams_dict["tower_depth"] + 1):
    tf_context_x = keras.layers.Dense(
        units=model_hyperParams_dict["tower_width"],
        activation="relu",
        name=f"context_dense_{layer_i}",
    )(tf_context_x)
    tf_context_x = keras.layers.Dropout(
        model_hyperParams_dict["dropout_percent"], name=f"context_dropOut_{layer_i}"
    )(tf_context_x)
tf_context_tower_output = keras.layers.Dense(
    units=model_hyperParams_dict["encoder_output_dim"],
    activation="relu",
    name=f"context_tower_output",
)(tf_context_x)


# combine outputs of the towers:
tf_tower_elementwise_prod = keras.layers.Multiply(name="tower_elementWise_product")(
    [tf_user_tower_output, tf_item_tower_output, tf_context_tower_output]
)
tf_tower_reduce_sum = tf.reduce_sum(
    tf_tower_elementwise_prod, axis=1, name="tf_tower_reduce_sum"
)
tf_model_output = tf.math.sigmoid(tf_tower_reduce_sum, name="model_output")

tf_2tower_model = keras.Model(
    inputs=[
        tf_user_ID_input,
        tf_user_feature_input,
        tf_item_ID_input,
        tf_item_feature_input,
        tf_context_feature_input,
    ],
    outputs=tf_model_output,
    name="TwoTowerModel",
)
keras.utils.plot_model(tf_2tower_model, show_shapes=True)

tf_2tower_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(
        learning_rate=model_hyperParams_dict["learning_rate"]
    ),
    metrics=[tf.keras.metrics.AUC()],
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    # stop training there is no improvement in validation loss for x epochs
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,  # return best weights found during training
)
train_history = tf_2tower_model.fit(
    x=[
        X_user_IDs.loc[:, :, :, "train"]["user_ID"].to_numpy(),
        X_user_attr_1hot.loc[:, :, :, "train"].to_numpy(),
        X_item_IDs.loc[:, :, :, "train"]["item_ID"].to_numpy(),
        X_item_attr_1hot.loc[:, :, :, "train"].to_numpy(),
        X_context_attr_1hot.loc[:, :, :, "train"].to_numpy(),
    ],
    y=model_data_df.loc[:, :, :, "train"]["bought"].to_numpy(),
    validation_data=[
        [
            X_user_IDs.loc[:, :, :, "validate"]["user_ID"].to_numpy(),
            X_user_attr_1hot.loc[:, :, :, "validate"].to_numpy(),
            X_item_IDs.loc[:, :, :, "validate"]["item_ID"].to_numpy(),
            X_item_attr_1hot.loc[:, :, :, "validate"].to_numpy(),
            X_context_attr_1hot.loc[:, :, :, "validate"].to_numpy(),
        ],
        model_data_df.loc[:, :, :, "validate"]["bought"].to_numpy(),
    ],
    epochs=model_hyperParams_dict["n_train_epochs"],
    verbose=1,
    callbacks=[early_stopping_callback],
    batch_size=model_hyperParams_dict["train_batch_size"],
)

plt.figure(figsize=(10, 5))
plt.plot(train_history.history["loss"], label="training data")
plt.plot(train_history.history["val_loss"], label="validation data")
plt.title("2-Tower Model Training (TensorFlow Implementation)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

# generate recommendations for a random user:
# (removing items already bought by user)
random_user_ID = X_user_IDs.sample(1)["user_ID"].values[0]
random_context = sim_obj.generate_random_context()
pred_df = item_attr_df.reset_index().copy()
for context_categ in include_featureNames_dict["context_feature_names"]:
    pred_df[context_categ] = random_context[context_categ]
pred_df["user_ID"] = random_user_ID
pred_df.set_index(["user_ID", "item_ID"], inplace=True)
pred_df = pred_df.join(user_attr_df, on="user_ID").reset_index()
pred_df = pred_df.merge(
    # note which items user has already bought:
    (
        model_data_df.loc[:, random_user_ID, :, :]
        .reset_index()
        .query("bought==1")[["item_ID", "bought"]]
        .drop_duplicates()
    ),
    how="left",
    on="item_ID",
)
# drop items already bought by user:
pred_df = pred_df.query("bought!=1").drop(["bought"], axis=1)
pred_df["pred_y"] = tf_2tower_model.predict(
    [
        pred_df["user_ID"].to_numpy(),
        sk_1hotEncoder_user_attr.transform(
            pred_df[include_featureNames_dict["user_feature_names"]]
        ),
        pred_df["item_ID"].to_numpy(),
        sk_1hotEncoder_item_attr.transform(
            pred_df[include_featureNames_dict["item_feature_names"]]
        ),
        sk_1hotEncoder_context_attr.transform(
            pred_df[include_featureNames_dict["context_feature_names"]]
        ),
    ]
)
pred_df["pred_y_rank"] = pred_df["pred_y"].rank(ascending=False).astype(int)
pred_df["true_y"] = [
    sim_obj.calc_user_preference_for_item(
        user_id=random_user_ID,
        item_id=ix,
        recommend_context=random_context,
    )["rating_in_this_context"]
    for ix in pred_df["item_ID"]
]
pred_df["true_y_rank"] = pred_df["true_y"].rank(ascending=False).astype(int)
pred_df.sort_values("pred_y", ascending=False, inplace=True)
plt.figure(figsize=(10, 5))
plt.scatter(pred_df["pred_y_rank"], pred_df["true_y_rank"], c=pred_df["pred_y_rank"])
plt.title(
    f"predicted vs. actual item affinity for user_ID={random_user_ID}\n(removed items already bought by user)"
)
plt.xlabel("predicted relative item affinity (rank)")
plt.ylabel("true relative item affinity (rank)")
