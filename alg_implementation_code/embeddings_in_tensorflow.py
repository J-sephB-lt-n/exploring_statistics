import numpy as np
from matplotlib import pyplot as plt

## Simulate Data ------------------------------------------------------------------------------------------------------------------------------------
n_users = 1_000
n_obs_per_user = 100

# define a 4x4 grid of response (y=1) probabilities
# each user in the data has a unique set of features (x1,x2)..
# ..which define which response (y=1) probability quadrant the user falls into
"""
                X2
                0-24    25-49   50-74   75-99
                0       1       2       3
X1  0-24    0   .       .       .       .
    25-49   1   .       .       .       .
    50-74   2   .       .       .       .
    75-99   3   .       .       .       .
"""
user_IDs = list(range(n_users))
probs_list = np.linspace(start=0, stop=1, num=16)
probs_grid = np.random.choice(probs_list, size=(4, 4), replace=False)
probs_grid = probs_grid.round(decimals=2)

# generate random features (x1,x2) for each user:
user_features_mat = (
    np.random.uniform(low=0, high=99, size=(n_users, 2)).round().astype(int)
)
# calculate true y=1 probability for each user:
quadrant_ref = np.floor(user_features_mat / 25).astype(int)
user_true_response_probs = np.array([probs_grid[q[0], q[1]] for q in quadrant_ref])

# simulate {n_obs_per_user} random observations per user:
X_mat = []
y_vec = []
for user_i in range(n_users):
    true_response_prob = user_true_response_probs[user_i]
    for x in range(n_obs_per_user):
        X_mat.append(
            [user_i]
        )  # the only input feature available to the model is the user ID
        y_vec.append(np.random.binomial(n=1, p=true_response_prob))
X_mat_train = np.array(X_mat)
y_vec_train = np.array(y_vec)

# shuffle the simulated data:
def unison_shuffled_copies(a, b):
    """
    this function stolen directly from Stack Overflow: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


X_mat_train, y_vec_train = unison_shuffled_copies(X_mat_train, y_vec_train)

## TensorFlow Implementation ------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")
tf.test.is_gpu_available()
tf.config.list_physical_devices()

tf_model_input = keras.Input(
    # shape is the shape of each input sample i.e. shape=(n_features,)
    # (the 2nd empty argument is the batch size - omitted here)
    shape=(1,),
    name="input_layer",
)
tf_layer_1 = keras.layers.Embedding(
    input_dim=n_users,  # size of the vocabulary i.e. n unique user IDs
    output_dim=2,  # desired dimension of the embeddings
    name="embedding_layer",
    input_length=1,  # this is used when the input to the embedding is a fixed-length sequence (e.g. a sentence containing multiple words)
)(tf_model_input)
tf_layer_1_flat = keras.layers.Flatten()(
    # flattening gets rid of the sequence dimension of the output of the embedding layer (the dimension is trivially 1 in this case)
    tf_layer_1
)
tf_layer_2 = keras.layers.Dense(units=20, activation="relu", name="dense_1")(
    tf_layer_1_flat
)
tf_layer_3 = keras.layers.Dense(units=20, activation="relu", name="dense_2")(tf_layer_2)
tf_layer_4 = keras.layers.Dense(units=20, activation="relu", name="dense_3")(tf_layer_3)
tf_layer_5 = keras.layers.Dense(units=20, activation="relu", name="dense_4")(tf_layer_4)
tf_layer_6 = keras.layers.Dense(units=20, activation="relu", name="dense_5")(tf_layer_5)
tf_model_output = keras.layers.Dense(
    units=1, activation="sigmoid", name="output_layer"
)(tf_layer_6)
tf_model = keras.Model(
    inputs=tf_model_input, outputs=tf_model_output, name="tensorflow_embedding_model"
)

tf_model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.01),
    metrics=["accuracy"],
)

tf_model.summary()
keras.utils.plot_model(tf_model)

train_history = tf_model.fit(X_mat_train, y_vec_train, batch_size=1_000, epochs=10)
tf_user_embeddings = tf_model.layers[1].get_weights()[0]

# plot the true probability quadrants that were used to generate the data:
fig = plt.figure(figsize=(10, 5))
plt.scatter(
    user_features_mat[:, 0],
    user_features_mat[:, 1],
    c=user_true_response_probs,
    cmap=plt.cm.Blues,
    s=20,
)
plt.title("Quadrants defining Pr[y=1] for Each User\n(in the Data Simulation Process)")
plt.colorbar(label="True Pr[y=1]")
plt.xlabel("x1")
plt.ylabel("x2")

# plot the user embeddings learned by the TensorFlow model:
fig = plt.figure(figsize=(10, 5))
plt.scatter(
    tf_user_embeddings[:, 0],
    tf_user_embeddings[:, 1],
    c=user_true_response_probs,
    cmap=plt.cm.Blues,
    s=20,
)
plt.title("User Embeddings Learned by TensorFlow Model")
plt.colorbar(label="True Pr[y=1]")
plt.xlabel("Embed. Dim. 1")
plt.ylabel("Embed. Dim. 2")
