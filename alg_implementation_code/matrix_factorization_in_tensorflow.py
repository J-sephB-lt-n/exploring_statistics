## simulate user/item data --------------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd

sys.path.append("..")
from recsys_simulation import recsys_data_simulator

sim_n_users = 10_000  # used by recsys_data_simulator
sim_n_items = 100  # used by recsys_data_simulator
sim_n_user_types = 10  # refer to help(recsys_data_simulator)

sim_obj = recsys_data_simulator(
    n_users=sim_n_users,
    n_items=sim_n_items,
    n_user_types=sim_n_user_types,
    n_mutations_per_user=5,
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
        "location": ["cape town", "london", "dubai", "new york", "rotterdam"],
        "age": ["infant", "teenager", "youth", "middle_aged", "old"],
    },
    potential_context_attr={
        "time_of_day": ["morning", "afternoon", "night"],
        "day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
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
    rating_range={"min": 1, "max": 10},
    rating_trunc_norm_std_dev=0.1,
)

sim_obj.expose_each_user_to_k_items(k=20)

obs_ratings_df = sim_obj.user_item_exposure_history_to_pandas_df()

## Ratings Matrix Factorization in Tensorflow -------------------------------------------------------------------------------------------------------
## define the matrix-factorisation model ##
import tensorflow as tf
import keras
from matplotlib import pyplot as plt

"""
-- NOTES --

In order for the TensorFlow embedding lookups to work:
        * user IDs must be encoded as a consecutive sequence of integers 0,1,2,3,...
        * item IDs must be encoded as a consecutive sequence of integers 0,1,2,3,... 
"""


class tensorflow_matrix_factorisation_model_class(keras.Model):
    def __init__(self, n_users, n_items, embed_dim, **kwargs):
        super(tensorflow_matrix_factorisation_model_class, self).__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embed_dim
        self.user_embedding = tf.keras.layers.Embedding(
            n_users, embed_dim, embeddings_initializer="he_normal"
        )
        self.item_embedding = tf.keras.layers.Embedding(
            n_items, embed_dim, embeddings_initializer="he_normal"
        )

    def call(self, inputs):
        """
        method documentation TODO
        """
        user_vectors = self.user_embedding(inputs[:, 0])
        item_vectors = self.item_embedding(inputs[:, 1])
        elementwise_prod = user_vectors * item_vectors
        dot_prods = tf.reduce_sum(elementwise_prod, axis=1)
        # force the the output to the rating range [1,10]:
        sigmoid_dot_prods = tf.math.sigmoid(dot_prods)
        return 1 + (sigmoid_dot_prods * 9)


tensorflow_matrix_factorisation_model = tensorflow_matrix_factorisation_model_class(
    n_users=sim_n_users,
    n_items=sim_n_items,
    embed_dim=10,
)

## define loss function and optimiser ##
tensorflow_matrix_factorisation_model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)

## each user contributes 1 observation to validation set and 1 observation to test set:
# (every user must appear in all data partitions: training, validation and test)
shuffle_obs_ratings_df = obs_ratings_df.sample(frac=1).copy()
pd_df_list = []
for user_id in sim_obj.user_dict:
    user_info = shuffle_obs_ratings_df.loc[user_id][["transaction_num"]]
    user_info["model_data_partition"] = np.random.choice(
        ["test"] + ["validate"] + ["train"] * (len(user_info) - 2),
        size=len(user_info),
        replace=False,
    )
    pd_df_list.append(user_info.copy())
model_data_partition_ref_df = pd.concat(pd_df_list, axis=0)

model_data_df = shuffle_obs_ratings_df.merge(
    model_data_partition_ref_df, how="left", on=["user_ID", "transaction_num"]
).reset_index()

## train the model ##
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    # stop training there is no improvement in validation loss for 3 epochs
    monitor="val_loss",
    patience=3,
)
train_history = tensorflow_matrix_factorisation_model.fit(
    x=model_data_df.query("model_data_partition=='train'")[
        ["user_ID", "item_ID"]
    ].values,
    y=model_data_df.query("model_data_partition=='train'")[
        "rounded_observed_rating"
    ].values,
    batch_size=1_000,
    epochs=100,
    verbose=1,
    validation_data=(
        model_data_df.query("model_data_partition=='validate'")[
            ["user_ID", "item_ID"]
        ].values,
        model_data_df.query("model_data_partition=='validate'")[
            "rounded_observed_rating"
        ].values,
    ),
    callbacks=[early_stopping_callback],
)

plt.plot(np.sqrt(train_history.history["loss"]), label="training data")
plt.plot(np.sqrt(train_history.history["val_loss"]), label="validation data")
plt.title("TensorFlow Matrix Factorization Model")
plt.ylabel("Loss (RMSE)")
plt.xlabel("Epoch")
plt.legend()

## generate predictions on test data ##
test_data_df = model_data_df.query("model_data_partition=='test'").copy()
test_preds = tensorflow_matrix_factorisation_model.predict(
    x=test_data_df[["user_ID", "item_ID"]].values
)
test_data_df["predicted_rating"] = test_preds

# plot distribution of predicted ratings (on test data)
plt.figure(figsize=(10, 5))
plt.hist(test_preds, bins=50)
plt.title("Distribution of Predicted Ratings (Test Set)")
plt.xlabel("Predicted Rating")
plt.ylabel("n Samples")

# plot scatterplot of predicted vs. true item affinity:
plt.figure(figsize=(10, 5))
plt.scatter(
    test_data_df["true_affinity_to_item"],
    test_data_df["predicted_rating"],
    s=2,
    alpha=0.2,
)
plt.title("Predicted vs. True Item Rating")
plt.xlabel("true item affinity (rating)")
plt.ylabel("predicted rating")
plt.axline([3, 3], [8, 8], color="black", alpha=0.2)

## generate 5 item recommendations for a random user ##
random_user_id = model_data_df.sample(1)["user_ID"].values[0]
exposed_item_IDs = (
    obs_ratings_df.loc[random_user_id].item_ID.drop_duplicates().values.tolist()
)
unexposed_item_IDs = [i for i in sim_obj.item_dict if i not in exposed_item_IDs]
random_user_all_unexposed_items_df = pd.DataFrame(
    {
        "user_ID": [random_user_id] * len(unexposed_item_IDs),
        "item_ID": unexposed_item_IDs,
    }
)
random_user_preds = tensorflow_matrix_factorisation_model.predict(
    x=random_user_all_unexposed_items_df[["user_ID", "item_ID"]].values
)
random_user_all_unexposed_items_df["predicted_rating"] = random_user_preds
k = 5
top_k_items_df = random_user_all_unexposed_items_df.sort_values(
    "predicted_rating", ascending=False
).head(k)
for item_idx, item_row in top_k_items_df.iterrows():
    sim_obj.expose_user_to_item(
        user_id=random_user_id, item_id=item_row.item_ID, log_interaction=False
    )
