## simulate ratings matrix data ##
import sys
import numpy as np

sys.path.append("..")
from recsys_simulation import recsys_data_simulator

sim_obj = recsys_data_simulator(
    n_users=1_000,
    n_items=100,
    n_user_types=10,
    n_mutations_per_user=5,
    potential_item_attr={
        "colour": ["red", "green", "blue", "black", "white"],
        "size": ["small", "medium", "large"],
        "material": ["metal", "wood", "cotton", "plastic"],
    },
    potential_user_attr={
        "location": ["cape town", "london", "dubai", "new york", "rotterdam"],
        "age": ["infant", "teenager", "youth", "middle_aged", "old"],
    },
    potential_context_attr={
        "time_of_day": ["morning", "afternoon", "night"],
        "day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    },
    rating_range={"min": 1, "max": 10},
    rating_trunc_norm_std_dev=1,
)
# expose each user to 5 unique items:
for user_i in sim_obj.user_dict.keys():
    random_item_ID_list = np.random.choice(
        list(sim_obj.item_dict.keys()), size=5, replace=False
    )
    for item_j in random_item_ID_list:
        sim_obj.expose_user_to_item(
            user_id=user_i, item_id=item_j, log_interaction=True
        )
