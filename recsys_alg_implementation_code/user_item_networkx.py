import sys
import networkx as nx
import walker  # for generating random walks quickly
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

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
sim_n_items = 200
min_buy_prob = 0
max_buy_prob = 0.1

sim_obj = recsys_data_simulator(
    n_users=sim_n_users,
    n_items=sim_n_items,
    n_user_types=5,
    n_item_attr_pref_mutations_per_user=5,
    n_additional_context_modifiers_per_user=1,
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
        "min": min_buy_prob,
        "max": max_buy_prob,
    },  # I use this "rating" as "item purchase probability"
    rating_trunc_norm_std_dev=0.01,
    n_context_effects=1,
    context_effect_abs_size=5,
)

pd.set_option("display.max_rows", sim_n_items)
pd.set_option("display.max_columns", 50)

# expose each user to the entire item catalogue:
# i.e. get the "item purchase probability" for each user/item combination
# (each exposure in a random context)
sim_obj.expose_each_user_to_k_items(
    min_k=sim_n_items,
    max_k=sim_n_items,
    ignore_context=True,
    add_noise_to_rating=False,  # this speeds up the function (and makes the patterns easier to model)
)

# decide (at random) which items each user buys (using item purchase probability):
for u in sim_obj.user_dict:
    for x in sim_obj.user_dict[u]["item_exposure_history"]:
        if np.random.uniform() < x["true_affinity_to_item"]:
            x["bought_item"] = 1
        else:
            x["bought_item"] = 0

# delete users without any item purchases:
users_to_delete_list = []
for u in sim_obj.user_dict:
    n_items_bought = sum(
        [x["bought_item"] for x in sim_obj.user_dict[u]["item_exposure_history"]]
    )
    if n_items_bought == 0:
        users_to_delete_list.append(u)
for u in users_to_delete_list:
    del sim_obj.user_dict[u]
print(f"{len(users_to_delete_list)} users deleted (no item purchases)")

# create a user-item graph, where an edge exists between a user and an item if the user bought that item
user_item_graph = nx.Graph()
for u in tqdm(sim_obj.user_dict, desc="creating user/item graph"):
    user_node_name = f"u{u}"
    user_item_graph.add_node(user_node_name, node_type="user")
    for x in sim_obj.user_dict[u]["item_exposure_history"]:
        item_node_name = f"i{x['item_ID']}"
        if item_node_name not in user_item_graph:
            user_item_graph.add_node(item_node_name, node_type="item")
        if x["bought_item"] == 1:
            user_item_graph.add_edge(user_node_name, item_node_name)

# plot the user/item graph:
# (plot is only readable if you have a small number of users and items)
if sim_n_users < 200:
    plt.figure(figsize=(20, 15))
    node_types_list = [
        user_item_graph.nodes[node]["node_type"]
        for node in list(user_item_graph.nodes())
    ]
    node_colours = [2 if x == "user" else 1 for x in node_types_list]
    nx.draw_networkx(user_item_graph, node_color=node_colours, cmap="Set1")

# in order to user the [walker](https://github.com/kerighan/graph-walker) library..
# ..nodes must have integer IDs:
user_item_graph_w_integer_labels = nx.convert_node_labels_to_integers(
    G=user_item_graph,
    label_attribute="str_node_name",  # store the original node name as an attribute of the node
)
# make a dictionary in order to be able to match users and items to their integer node IDs:
int_node_label_ref_dict = {}
for node_id in user_item_graph_w_integer_labels.nodes:
    str_node_name = user_item_graph_w_integer_labels.nodes[node_id]
    int_node_label_ref_dict[node_id] = str_node_name
    int_node_label_ref_dict[str_node_name["str_node_name"]] = node_id

# generate item recommendations for a random user by using random walks directly --------------------------------------------------------------------
n_item_recs_per_user = 20
random_user_ID = np.random.choice(list(sim_obj.user_dict.keys()))
random_user_node_name = f"u{random_user_ID}"
random_user_int_node_ID = int_node_label_ref_dict[random_user_node_name]
items_already_bought_df = pd.DataFrame(
    {
        "user_ID": random_user_ID,
        "item_ID": [
            x["item_ID"]
            for x in sim_obj.user_dict[random_user_ID]["item_exposure_history"]
            if x["bought_item"] == 1
        ],
        "bought_item": 1,
    }
)
true_pref_df_all_items = pd.DataFrame(
    {
        "user_ID": random_user_ID,
        "item_ID": list(sim_obj.item_dict.keys()),
        "true_buy_prob": [
            sim_obj.calc_user_preference_for_item(
                user_id=random_user_ID,
                item_id=i,
                recommend_context=sim_obj.generate_random_context(),
            )["rating_ignore_context"]
            for i in list(sim_obj.item_dict.keys())
        ],
    }
).merge(items_already_bought_df, on=["user_ID", "item_ID"], how="left")
true_pref_df_all_items.loc[
    true_pref_df_all_items["bought_item"].isna(), "bought_item"
] = 0
true_pref_df_all_items["bought_item"] = true_pref_df_all_items["bought_item"].astype(
    int
)
random_walks = walker.random_walks(
    user_item_graph, n_walks=10_000, walk_len=4, start_nodes=[random_user_int_node_ID]
)
items_collected_on_random_walks = np.unique(
    np.array(
        [random_walks[:, j] for j in range(1, random_walks.shape[1], 2)]
    ).flatten(),
    return_counts=True,
)
items_to_recommend_df = pd.DataFrame(
    {
        "user_ID": random_user_ID,
        "item_int_node_ID": items_collected_on_random_walks[0],
        "item_ID": [
            int(int_node_label_ref_dict[i]["str_node_name"][1:])
            for i in items_collected_on_random_walks[0]
        ],
        "n_times_visited_random_walk": items_collected_on_random_walks[1],
    }
)
true_pref_df_all_items = true_pref_df_all_items.merge(
    items_to_recommend_df[["user_ID", "item_ID", "n_times_visited_random_walk"]],
    on=["user_ID", "item_ID"],
    how="left",
)
true_pref_df_all_items["random_seed"] = np.random.uniform(
    size=len(true_pref_df_all_items)
)
true_pref_df_all_items.sort_values(
    ["n_times_visited_random_walk", "random_seed"], ascending=False, inplace=True
)
new_items_to_recommend = true_pref_df_all_items.query("bought_item==0").head(
    n_item_recs_per_user
)[["user_ID", "item_ID"]]
new_items_to_recommend["randomWalk_recommend_item"] = 1
true_pref_df_all_items = true_pref_df_all_items.merge(
    new_items_to_recommend, on=["user_ID", "item_ID"], how="left"
)
true_pref_df_all_items.loc[
    true_pref_df_all_items["randomWalk_recommend_item"].isna(),
    "randomWalk_recommend_item",
] = 0
true_pref_df_all_items["randomWalk_recommend_item"] = true_pref_df_all_items[
    "randomWalk_recommend_item"
].astype(int)

fig, axs = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 5)
)  # note: figsize is the size of the global plot
axs[0].hist(
    x=true_pref_df_all_items.query(
        "bought_item==0 & randomWalk_recommend_item==1"
    ).true_buy_prob,
    range=[min_buy_prob, max_buy_prob],
    bins=20,
)
axs[0].set_xlabel("true buy probability (unknown to model)")
axs[0].set_ylabel("n items")
axs[0].set_title(f"Recommended Items ({n_item_recs_per_user} items)")
axs[1].hist(
    x=true_pref_df_all_items.query(
        "bought_item==0 & randomWalk_recommend_item==0"
    ).true_buy_prob,
    range=[min_buy_prob, max_buy_prob],
    bins=20,
)
axs[1].set_xlabel("true buy probability (unknown to model)")
axs[1].set_ylabel("n items")
axs[1].set_title(
    f"Items NOT Recommended ({len(true_pref_df_all_items.query('bought_item==0 & randomWalk_recommend_item==0'))} items)"
)
fig.suptitle(
    f"Item Recommendations for user_ID={random_user_ID}\n(based on random walk on user/item graph)\n(only considering items not already bought)"
)
fig.tight_layout()

# generate item recommendations for the same random user using their user-neighbourbood defined on the graph ------------------------------------------------
n_users_in_neighHood = 100
random_walks = walker.random_walks(
    user_item_graph, n_walks=10_000, walk_len=5, start_nodes=[random_user_int_node_ID]
)
users_collected_on_random_walks = np.unique(
    np.array(
        [random_walks[:, j] for j in range(0, random_walks.shape[1], 2)]
    ).flatten(),
    return_counts=True,
)
collected_potential_neighbours_df = pd.DataFrame(
    {
        "user_ID": random_user_ID,
        "neighb_user_int_node_ID": users_collected_on_random_walks[0],
        "neighb_user_ID": [
            # (int_node_label_ref_dict[u]["str_node_name"], int_node_label_ref_dict[u]["str_node_name"][1:])
            int(int_node_label_ref_dict[u]["str_node_name"][1:])
            for u in users_collected_on_random_walks[0]
        ],
        "n_visits_on_random_walk": users_collected_on_random_walks[1],
        "random_seed": np.random.uniform(size=len(users_collected_on_random_walks[0])),
    }
).query("user_ID!=neighb_user_ID")
neighbours_df = collected_potential_neighbours_df.sort_values(
    ["n_visits_on_random_walk", "random_seed"], ascending=False
).head(n_users_in_neighHood)
neighbours_df["neighb_weight"] = (
    neighbours_df.n_visits_on_random_walk / neighbours_df.n_visits_on_random_walk.sum()
)
# get bought/not_bought status of each item, for every neighbour:
store_pd_df_list = []
for neighb_user_ID in tqdm(
    neighbours_df.neighb_user_ID, desc="neighbour true item buy probs -> pd.DataFrame"
):
    for x in sim_obj.user_dict[neighb_user_ID]["item_exposure_history"]:
        store_pd_df_list.append(
            pd.DataFrame(
                {
                    "neighb_user_ID": [neighb_user_ID],
                    "item_ID": [x["item_ID"]],
                    "bought_item": [x["bought_item"]],
                }
            )
        )
neighbours_item_buy_hist_df = pd.concat(store_pd_df_list, axis=0)
neighbours_item_buy_hist_df = neighbours_item_buy_hist_df.merge(
    neighbours_df[["neighb_user_ID", "n_visits_on_random_walk", "neighb_weight"]],
    on="neighb_user_ID",
    how="left",
)
neighbours_item_buy_hist_df["buy_vote"] = (
    neighbours_item_buy_hist_df["bought_item"]
    * neighbours_item_buy_hist_df["neighb_weight"]
)
neighb_recommend_calc_df = (
    neighbours_item_buy_hist_df.groupby("item_ID")
    .agg(
        unweighted_neighb_vote=("bought_item", "mean"),
        weighted_neighb_vote=("buy_vote", "sum"),
    )
    .sort_values("weighted_neighb_vote", ascending=False)
)
true_pref_df_all_items = true_pref_df_all_items.merge(
    neighb_recommend_calc_df.reset_index(), on="item_ID", how="left"
)
new_items_to_recommend = (
    true_pref_df_all_items.query("bought_item==0")
    .sort_values(["weighted_neighb_vote", "random_seed"], ascending=False)
    .head(n_item_recs_per_user)
)[["user_ID", "item_ID"]]
new_items_to_recommend["wtd_neighHood_recommend_item"] = 1
true_pref_df_all_items = true_pref_df_all_items.merge(
    new_items_to_recommend, on=["user_ID", "item_ID"], how="left"
)
true_pref_df_all_items.loc[
    true_pref_df_all_items["wtd_neighHood_recommend_item"].isna(),
    "wtd_neighHood_recommend_item",
] = 0
true_pref_df_all_items["wtd_neighHood_recommend_item"] = true_pref_df_all_items[
    "wtd_neighHood_recommend_item"
].astype(int)

fig, axs = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 5)
)  # note: figsize is the size of the global plot
axs[0].hist(
    x=true_pref_df_all_items.query(
        "bought_item==0 & wtd_neighHood_recommend_item==1"
    ).true_buy_prob,
    range=[min_buy_prob, max_buy_prob],
    bins=20,
    color="green",
)
axs[0].set_xlabel("true buy probability (unknown to model)")
axs[0].set_ylabel("n items")
axs[0].set_title(f"Recommended Items ({n_item_recs_per_user} items)")
axs[1].hist(
    x=true_pref_df_all_items.query(
        "bought_item==0 & wtd_neighHood_recommend_item==0"
    ).true_buy_prob,
    range=[min_buy_prob, max_buy_prob],
    bins=20,
    color="green",
)
axs[1].set_xlabel("true buy probability (unknown to model)")
axs[1].set_ylabel("n items")
axs[1].set_title(
    f"Items NOT Recommended ({len(true_pref_df_all_items.query('bought_item==0 & randomWalk_recommend_item==0'))} items)"
)
fig.suptitle(
    f"Item Recommendations for user_ID={random_user_ID}\n(based on user neighbourhood on user/item graph - weighted vote of {n_users_in_neighHood} neighbours)\n(only considering items not already bought)"
)
fig.tight_layout()
