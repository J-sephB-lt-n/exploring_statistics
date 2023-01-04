## simulate user/item data ##
import sys
import numpy as np

sys.path.append("..")
from recsys_simulation import recsys_data_simulator

sim_n_users = 1_000
sim_n_items = 200

sim_obj = recsys_data_simulator(
    n_users=sim_n_users,
    n_items=sim_n_items,
    n_user_types=10,
    n_item_attr_pref_mutations_per_user=10,
    n_additional_context_modifiers_per_user=2,
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
        "age_group": ["infant", "teenager", "youth", "middle_aged", "old"],
        "affluence": ["low", "middle", "high"],
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
    rating_range={"min": 0, "max": 0.25},
    rating_trunc_norm_std_dev=0.01,
    n_context_effects=2,
    context_effect_abs_size=5,
)
# expose each user to the entire item catalogue:
sim_obj.expose_each_user_to_k_items(
    min_k=sim_n_items,
    max_k=sim_n_items,
)


## item-wise rules --------------------------------------------------------------------------------
# example: {bread, tomato}=>{cheese}
# i.e. users who have bought bread and tomato should be recommended cheese
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# get data into format expected by mlxtend:
pd_df_list = []
for user_id in sim_obj.user_dict:
    item_exposure_history = sim_obj.user_dict[user_id]["item_exposure_history"]
    item_vec = [0] * len(sim_obj.item_dict)
    for transaction in item_exposure_history:
        if transaction["rounded_observed_rating"] == 1:
            # only record transactions where user_rating=1
            # (i.e. we are generating implicit ratings data)
            item_vec[transaction["item_ID"]] = 1
    pd_df_list.append(
        pd.DataFrame(
            [[user_id] + item_vec],
            columns=["user_ID"] + [f"item_{i}" for i in range(len(item_vec))],
        )
    )
item_item_df = pd.concat(pd_df_list, axis=0).set_index("user_ID")
frequent_itemsets = apriori(
    item_item_df.astype(bool),
    min_support=0.002,  # itemset must appear at least min_support% of total baskets
    use_colnames=False,
)
item_item_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
"""
Antecedent Support: % of users who bought the antecedent 
Consequent Support: % of users who bought the consequent   
Support:            % of users who bought both the antecedent and the consequent
Confidence:         (n users bought both antecedent & consequent) / (n users bought antecedent)                    
Lift:               (% of users bought both antecedent & consequent) / ( (% of users bought antecedent) * (% of users bought consequent) ) 
Leverage:           ?
Conviction:         ?
"""
item_item_rules.sort_values("confidence", ascending=False)
item_item_rules.sort_values("lift", ascending=False)

# generate recommendations for a random user:
random_user_id = np.random.choice(list(sim_obj.user_dict.keys()))
user_items_bought_history = np.where(item_item_df.loc[random_user_id].values > 0)[
    0
].tolist()
rules_applicable_to_user = []
# gather all rules applying to user:
for rule_idx, rule in item_item_rules.iterrows():
    if len(rule.antecedents.intersection(set(user_items_bought_history))) == len(
        rule.antecedents
    ):
        # if full antecedent itemset is in user buy history:
        consequent_itemset = rule.consequents
        # remove items in the consequent item set that the user has already bought:
        recommendable_consequent_itemset = consequent_itemset.difference(
            set(user_items_bought_history)
        )
        if len(recommendable_consequent_itemset) > 0:
            rules_applicable_to_user.append(rule_idx)

# get top k rules applying to user (highest lift rules):
k = 3
top_k_rules = (
    item_item_rules.loc[rules_applicable_to_user]
    .sort_values("lift", ascending=False)
    .head(k)
)

for rule_idx, rule in top_k_rules.iterrows():
    # if full antecedent itemset is in user buy history:
    consequent_itemset = rule.consequents
    # remove items in the consequent item set that the user has already bought:
    recommendable_consequent_itemset = consequent_itemset.difference(
        set(user_items_bought_history)
    )

    print(
        f"""
            of the {int(rule['antecedent support']*sim_n_users)} customers who bought item(s) {list(rule.antecedents)}, 
                {int(rule.support*sim_n_users)} ({100*rule.confidence:.0f}%) also bought item(s) {list(rule.consequents)} 
            rule lift:          {rule.lift:.3f}
        """
    )
    for recommend_item_id in recommendable_consequent_itemset:
        sim_obj.expose_user_to_item(
            user_id=random_user_id,
            item_id=recommend_item_id,
            log_interaction=False,
        )
