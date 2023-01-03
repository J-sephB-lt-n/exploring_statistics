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

## Ratings Matrix Factorization in PyTorch -------------------------------------------------------------------------------------------------------
## define the matrix-factorisation model ##
"TODO"
