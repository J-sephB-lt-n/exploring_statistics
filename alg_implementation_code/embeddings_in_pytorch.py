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

## PyTorch Implementation ------------------------------------------------------------------------------------------------------------------------
import torch
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class PyTorchEmbedNetwork(nn.Module):
    def __init__(self):
        super(
            PyTorchEmbedNetwork, self
        ).__init__()  # this allows us to inherit from the parent class (nn.Module)
        self.embedding_lookup = nn.Embedding(
            num_embeddings=n_users,
            embedding_dim=2,
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=1),
        )

    def forward(self, x):
        user_embed = self.embedding_lookup(x)
        logits = self.linear_relu_stack(user_embed)
        logits = torch.flatten(logits)
        return nn.Sigmoid()(logits)


pt_model = PyTorchEmbedNetwork()
print(pt_model)

pt_loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
pt_optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.01)

# put data into a PyTorch-friendly format:
class train_data_torch_dataset(torch.utils.data.Dataset):
    """A torch dataset class is used to store data in a format that PyTorch can interact with"""

    def __init__(self, X_matrix, y_vec):
        super(
            train_data_torch_dataset, self
        ).__init__()  # this allows us to inherit from the parent class (torch.utils.data.Dataset)
        self.X_matrix = X_matrix
        self.y_vec = y_vec

    def __len__(self):
        return len(self.y_vec)

    def __getitem__(self, idx):
        X_vec = self.X_matrix[idx, :]
        y_scalar = self.y_vec[idx]

        return (
            torch.tensor(X_vec, dtype=torch.int),
            torch.tensor(y_scalar, dtype=torch.int),
        )


pt_train_data = train_data_torch_dataset(X_matrix=X_mat_train, y_vec=y_vec_train)
pt_data_loader = torch.utils.data.DataLoader(
    dataset=pt_train_data, batch_size=1_000, shuffle=True
)

loss_history_per_epoch = []

n_epochs = 10

for epoch in range(1, n_epochs + 1):
    print(f"started epoch {epoch}")
    for X, y in pt_data_loader:
        # Set gradients to zero (otherwise pytorch accumulates gradients from previous iterations)
        pt_optimizer.zero_grad()

        model_predictions = pt_model(X)
        loss = pt_loss_fn(model_predictions, y.float())

        # Backpropagate
        loss.backward()

        # Update the parameters
        pt_optimizer.step()

    print(f"completed epoch {epoch}")
    print(f"Training Loss: {loss.item()}")
    loss_history_per_epoch.append(loss.item())

plt.figure(figsize=(10, 5))
plt.plot(range(len(loss_history_per_epoch)), loss_history_per_epoch)
# plt.plot(range(len(loss_history_per_epoch)), np.cumsum(loss_history_per_epoch)/np.array(range(1,len(loss_history_per_epoch)+1)))  # cumulative mean loss
plt.title("Loss (BCE) on Training Data")
plt.xlabel("epoch")
plt.ylabel("Loss (BCE)")

pt_user_embeddings = (
    pt_model.embedding_lookup(torch.tensor(user_IDs)).cpu().detach().numpy()
)

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

# plot the user embeddings learned by the PyTorch model:
fig = plt.figure(figsize=(10, 5))
plt.scatter(
    pt_user_embeddings[:, 0],
    pt_user_embeddings[:, 1],
    c=user_true_response_probs,
    cmap=plt.cm.Blues,
    s=20,
)
plt.title("User Embeddings Learned by PyTorch Model")
plt.colorbar(label="True Pr[y=1]")
plt.xlabel("Embed. Dim. 1")
plt.ylabel("Embed. Dim. 2")
