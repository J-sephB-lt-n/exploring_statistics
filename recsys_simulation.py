class recsys_data_simulator:
    """
    This class can be used to simulate data to use for testing recommendation models
    The simulated user population is defined by a small number of user "stereotypes"..
        ..each "stereotype" is defined by a particular set of user_attributes and a mix of item attribute preferences
    Each individual user is created by selecting (and then mutating) one of these "stereotypes"
        (i.e. each simulated user differs slightly - at random - from their base type)
    When a particular user is presented with a particular item..
        ..the user will draw a rating at random for that item..
        ..from a truncated normal distribution (truncated at rating_range["min"] and rating_range["max"])..
        ..where the mean of this distribution is determined by a comparison of the item's attributes with the user's preferences

    Examples
    ----------
    # an example illustrating the full data simulation process (with extensive text documentation) is available within the class itself:
    # run in python:
    >> from recsys_simulation import recsys_data_simulator
    >> sim_obj = recsys_data_simulator()        # using default settings simulates a small toy user/item population ideal for the generate_tutorial() method
    >> print( sim_obj.generate_tutorial() )     # note that the tutorial uses the population defined in the recsys_data_simulator class
    ...

    Methods
    ----------
    generate_random_context
        documentation TODO
    calc_user_preference_for_item
        documentation TODO
    expose_user_to_item
        documentation TODO
    expose_each_user_to_k_items
        documentation TODO
    user_item_exposure_history_to_pandas_df
        documentation TODO
    user_attr_data_to_pandas_df
        documentation TODO
    item_attr_data_to_pandas_df
        documentation TODO
    generate_tutorial
        documentation TODO
    """

    def __init__(
        self,
        n_users=5,
        n_items=3,
        n_user_types=3,
        n_item_attr_pref_mutations_per_user=2,
        n_additional_context_modifiers_per_user=2,
        potential_item_attr={
            "colour": [
                "black",
                "white",
            ],
            "size": ["small", "medium", "large"],
            "material": [
                "metal",
                "wood",
                "cotton",
                "plastic",
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
        },
        rating_range={
            "min": 1,
            "max": 10,
        },
        rating_trunc_norm_std_dev=0.2,
        n_context_effects=2,
        context_effect_abs_size=2,
    ):
        """
        method documentation TODO
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_types = n_user_types
        self.n_item_attr_pref_mutations_per_user = n_item_attr_pref_mutations_per_user
        self.n_additional_context_modifiers_per_user = (
            n_additional_context_modifiers_per_user
        )
        self.potential_item_attr = potential_item_attr
        self.potential_user_attr = potential_user_attr
        self.potential_context_attr = potential_context_attr
        self.rating_range = rating_range
        self.rating_trunc_norm_std_dev = rating_trunc_norm_std_dev
        self.n_context_effects = n_context_effects
        self.context_effect_abs_size = context_effect_abs_size
        self.item_dict = {}
        self.user_types_dict = {}
        self.user_dict = {}
        self.max_achievable_preference_score = 10 * len(
            self.potential_item_attr
        )  # the highest achievable user preference for an item

        assert (
            rating_range["max"] > rating_range["min"]
        ), "must have rating_range['max'] > rating_range['min']"

        import copy
        import numpy as np
        import sys

        # set up progress bar:
        print(
            f"simulating items |{' '*25}| 0%", flush=True, file=sys.stdout, end="\r"
        )  # start of progress bar
        sim_item_counter = 0  # used by progress bar
        n_completed_symbols = 0  # used by progress bar
        for i in range(self.n_items):
            self.item_dict[i] = {}
            for attr in self.potential_item_attr:
                self.item_dict[i][attr] = np.random.choice(
                    self.potential_item_attr[attr]
                )
            sim_item_counter += 1

            # print updated progress bar:
            completed_percent = sim_item_counter / self.n_items
            update_n_completed_symbols = int(100 * completed_percent / 4)
            if update_n_completed_symbols > n_completed_symbols:
                # only update progress bar if there is enough completion to print a new symbol
                n_completed_symbols = update_n_completed_symbols
                n_uncompleted_symbols = 25 - n_completed_symbols
                print(
                    f"simulating items |{'.'*n_completed_symbols}{' '*n_uncompleted_symbols}| {n_completed_symbols*4}%",
                    flush=True,
                    file=sys.stdout,
                    end="\r",
                )
        print("")  # to keep completed progress bar printed in terminal

        # simulate user types:
        # set up progress bar:
        print(
            f"simulating user types |{' '*25}| 0%",
            flush=True,
            file=sys.stdout,
            end="\r",
        )  # start of progress bar
        sim_user_type_counter = 0  # used by progress bar
        n_completed_symbols = 0  # used by progress bar
        for i in range(n_user_types):
            self.user_types_dict[i] = {}
            for user_attr in potential_user_attr:
                self.user_types_dict[i][user_attr] = np.random.choice(
                    potential_user_attr[user_attr]
                )
            self.user_types_dict[i]["relative_item_attr_preferences"] = {}
            for item_attr in self.potential_item_attr:
                self.user_types_dict[i]["relative_item_attr_preferences"][
                    item_attr
                ] = {}
                for val in self.potential_item_attr[item_attr]:
                    self.user_types_dict[i]["relative_item_attr_preferences"][
                        item_attr
                    ][val] = np.random.randint(low=0, high=10)
            self.user_types_dict[i]["context_modifiers"] = {}
            for context_category in self.potential_context_attr:
                self.user_types_dict[i]["context_modifiers"][context_category] = {}
                for context_val in self.potential_context_attr[context_category]:
                    self.user_types_dict[i]["context_modifiers"][context_category][
                        context_val
                    ] = []
                    for x in range(self.n_context_effects):
                        random_item_attr = np.random.choice(
                            list(self.potential_item_attr.keys())
                        )
                        random_item_attr_val = np.random.choice(
                            self.potential_item_attr[random_item_attr]
                        )
                        self.user_types_dict[i]["context_modifiers"][context_category][
                            context_val
                        ].append(
                            (
                                random_item_attr,
                                random_item_attr_val,
                                np.random.choice([-1, 1])
                                * self.context_effect_abs_size,
                            )
                        )
            sim_user_type_counter += 1
            # print updated progress bar:
            completed_percent = sim_user_type_counter / self.n_user_types
            update_n_completed_symbols = int(100 * completed_percent / 4)
            if update_n_completed_symbols > n_completed_symbols:
                # only update progress bar if there is enough completion to print a new symbol
                n_completed_symbols = update_n_completed_symbols
                n_uncompleted_symbols = 25 - n_completed_symbols
                print(
                    f"simulating user types |{'.'*n_completed_symbols}{' '*n_uncompleted_symbols}| {n_completed_symbols*4}%",
                    flush=True,
                    file=sys.stdout,
                    end="\r",
                )
        print("")  # to keep completed progress bar printed in terminal

        ## simulate population of users ##
        # set up progress bar:
        print(
            f"simulating users |{' '*25}| 0%", flush=True, file=sys.stdout, end="\r"
        )  # start of progress bar
        sim_user_counter = 0  # used by progress bar
        n_completed_symbols = 0  # used by progress bar
        for i in range(n_users):
            random_user_type_id = np.random.choice(range(self.n_user_types))
            self.user_dict[i] = copy.deepcopy(self.user_types_dict[random_user_type_id])
            self.user_dict[i]["item_exposure_history"] = []
            self.user_dict[i]["user_type_ID"] = random_user_type_id
            self.user_dict[i]["mutations_log"] = {
                "item_attribute_preferences": [],
                "context_modifiers": [],
            }
            for _ in range(self.n_item_attr_pref_mutations_per_user):
                random_item_attr = np.random.choice(
                    list(self.potential_item_attr.keys())
                )
                random_val = np.random.choice(
                    self.potential_item_attr[random_item_attr]
                )
                prev_value = self.user_dict[i]["relative_item_attr_preferences"][
                    random_item_attr
                ][random_val]
                new_value = np.random.randint(low=0, high=10)
                self.user_dict[i]["mutations_log"]["item_attribute_preferences"].append(
                    (random_item_attr, random_val, f"{prev_value}->{new_value}")
                )
                self.user_dict[i]["relative_item_attr_preferences"][random_item_attr][
                    random_val
                ] = new_value
            # add additional context modifiers to the user:
            for context_category in self.potential_context_attr:
                for context_val in self.potential_context_attr[context_category]:
                    for _ in range(self.n_additional_context_modifiers_per_user):
                        random_item_attr = np.random.choice(
                            list(self.potential_item_attr.keys())
                        )
                        random_item_attr_val = np.random.choice(
                            self.potential_item_attr[random_item_attr]
                        )
                        context_modifier = (
                            random_item_attr,
                            random_item_attr_val,
                            np.random.choice([-1, 1]) * self.context_effect_abs_size,
                        )
                        self.user_dict[i]["context_modifiers"][context_category][
                            context_val
                        ].append(context_modifier)
                        self.user_dict[i]["mutations_log"]["context_modifiers"].append(
                            (context_category, context_val, context_modifier)
                        )

            sim_user_counter += 1  # used by progress bar

            # print updated progress bar:
            completed_percent = sim_user_counter / n_users
            update_n_completed_symbols = int(100 * completed_percent / 4)
            if update_n_completed_symbols > n_completed_symbols:
                # only update progress bar if there is enough completion to print a new symbol
                n_completed_symbols = update_n_completed_symbols
                n_uncompleted_symbols = 25 - n_completed_symbols
                print(
                    f"simulating users |{'.'*n_completed_symbols}{' '*n_uncompleted_symbols}| {n_completed_symbols*4}%",
                    flush=True,
                    file=sys.stdout,
                    end="\r",
                )
        print("")  # to keep completed progress bar printed in terminal

    def generate_random_context(self):
        """
        TODO: method documentation here
        """
        import numpy as np

        random_recommend_context = {}
        for context_categ in self.potential_context_attr:
            random_recommend_context[context_categ] = np.random.choice(
                self.potential_context_attr[context_categ]
            )

        return random_recommend_context

    def calc_user_preference_for_item(self, user_id, item_id, recommend_context):
        """
        This function calculates the affinity score for a specified user/item/(context) combination
        i.e. it quantifies how much does user [user_id] likes item [item_id] in context [recommend_context] (context optional)
        The returned value is rescaled to lie in the range [rating_range]
        """
        import copy

        item_attr_info = self.item_dict[item_id]
        user_pref_info = self.user_dict[user_id]["relative_item_attr_preferences"]

        # adjust user item preferences based on the [recommend_context]:
        user_context_mods = self.user_dict[user_id]["context_modifiers"]
        context_adj_user_pref_info = copy.deepcopy(user_pref_info)
        for context_categ in recommend_context:
            context_mods = user_context_mods[context_categ][
                recommend_context[context_categ]
            ]
            for context_mod in context_mods:
                item_attr = context_mod[0]
                item_attr_val = context_mod[1]
                mod_val = context_mod[2]
                adj_user_pref_val = context_adj_user_pref_info[item_attr][item_attr_val]
                if (adj_user_pref_val + mod_val) <= self.rating_range["max"] and (
                    adj_user_pref_val + mod_val
                ) >= self.rating_range["min"]:
                    # if the adjusted rating still falls into the rating range
                    context_adj_user_pref_info[item_attr][item_attr_val] += mod_val

        raw_affinity_score = 0
        context_adj_raw_affinity_score = 0
        for attr in item_attr_info:
            attr_val = item_attr_info[attr]
            raw_affinity_score += user_pref_info[attr][attr_val]
            context_adj_raw_affinity_score += context_adj_user_pref_info[attr][attr_val]

        # scale raw score to lie in the range {self.rating_range}
        scaled_score = (
            self.rating_range["min"]
            + (
                raw_affinity_score
                * (self.rating_range["max"] - self.rating_range["min"])
            )
            / self.max_achievable_preference_score
        )

        context_adj_scaled_score = (
            self.rating_range["min"]
            + (
                context_adj_raw_affinity_score
                * (self.rating_range["max"] - self.rating_range["min"])
            )
            / self.max_achievable_preference_score
        )

        return {
            "rating_ignore_context": scaled_score,
            "rating_in_this_context": context_adj_scaled_score,
        }

    def expose_user_to_item(
        self,
        user_id,
        item_id,
        recommend_context,
        ignore_context,
        log_interaction,
        add_noise_to_rating,
    ):
        """
        method documentation TODO
        """

        true_user_affinity_to_item = self.calc_user_preference_for_item(
            user_id=user_id,
            item_id=item_id,
            recommend_context=recommend_context,
        )
        if ignore_context:
            true_user_affinity_to_item = true_user_affinity_to_item[
                "rating_ignore_context"
            ]
        else:
            true_user_affinity_to_item = true_user_affinity_to_item[
                "rating_in_this_context"
            ]

        if add_noise_to_rating:
            from scipy.stats import truncnorm

            observed_user_rating = (
                truncnorm(
                    (self.rating_range["min"] - true_user_affinity_to_item)
                    / self.rating_trunc_norm_std_dev,
                    (self.rating_range["max"] - true_user_affinity_to_item)
                    / self.rating_trunc_norm_std_dev,
                    loc=true_user_affinity_to_item,
                    scale=self.rating_trunc_norm_std_dev,
                )
                .rvs(1)
                .tolist()[0]
            )
        else:
            observed_user_rating = true_user_affinity_to_item

        interaction_result = {
            "item_ID": item_id,
            "true_affinity_to_item": true_user_affinity_to_item,
            "observed_rating": observed_user_rating,
            "rounded_observed_rating": round(observed_user_rating),
            "recommend_context": recommend_context,
        }

        if log_interaction:
            self.user_dict[user_id]["item_exposure_history"].append(interaction_result)
        else:
            if ignore_context:
                context_msg = "(recommendation context ignored)"
            else:
                context_msg = ""
            print(
                f"""exposed user {user_id} to item {item_id}:   observed rating was {observed_user_rating:.3f}
                        {context_msg}
                        (interaction not logged)
                """
            )

    def expose_each_user_to_k_items(
        self, min_k, max_k, ignore_context, add_noise_to_rating
    ):
        """
        Each user is exposed to [k] r random items, where [k] is an integer in [min_k, max_k]

        TODO: method documentation here
        """

        import sys
        import numpy as np

        # set up progress bar:
        print(
            f"exposing users to random items |{' '*25}| 0%",
            flush=True,
            file=sys.stdout,
            end="\r",
        )  # start of progress bar
        sim_user_counter = 1  # used by progress bar
        n_completed_symbols = 0  # used by progress bar
        for user_i in self.user_dict.keys():
            random_item_ID_list = np.random.choice(
                list(self.item_dict.keys()),
                size=np.random.choice(range(min_k, max_k + 1)),
                replace=False,
            )
            for item_j in random_item_ID_list:
                self.expose_user_to_item(
                    user_id=user_i,
                    item_id=item_j,
                    log_interaction=True,
                    ignore_context=ignore_context,
                    add_noise_to_rating=add_noise_to_rating,
                    recommend_context=self.generate_random_context(),
                )
            sim_user_counter += 1  # used by progress bar
            # print updated progress bar:
            completed_percent = sim_user_counter / (self.n_users)
            update_n_completed_symbols = int(100 * completed_percent / 4)
            if update_n_completed_symbols > n_completed_symbols:
                # only update progress bar if there is enough completion to print a new symbol
                n_completed_symbols = update_n_completed_symbols
                n_uncompleted_symbols = 25 - n_completed_symbols
                print(
                    f"exposing users to random items |{'.'*n_completed_symbols}{' '*n_uncompleted_symbols}| {n_completed_symbols*4}%",
                    flush=True,
                    file=sys.stdout,
                    end="\r",
                )
        print("")  # to keep completed progress bar printed in terminal

    def user_item_exposure_history_to_pandas_df(self):
        """
        method documentation TODO
        """
        print("exporting user/item exposure history to pandas.DataFrame: ", end="")
        import pandas as pd

        pd_df_list = []
        for user_id in self.user_dict:
            transaction_counter = 1
            for transaction in self.user_dict[user_id]["item_exposure_history"]:
                pd_df_list.append(
                    pd.DataFrame(
                        {
                            "transaction_num": [transaction_counter],
                            "item_ID": [transaction["item_ID"]],
                            "rounded_observed_rating": [
                                transaction["rounded_observed_rating"]
                            ],
                            "observed_rating": [transaction["observed_rating"]],
                            "true_affinity_to_item": [
                                transaction["true_affinity_to_item"]
                            ],
                        },
                        index=[user_id],
                    )
                )
                transaction_counter += 1
        if len(pd_df_list) > 0:
            result_df = pd.concat(pd_df_list, axis=0)
            result_df.index.name = "user_ID"
        else:
            print("there are no historic user/item interactions")
            result_df = None

        print("...done")
        return result_df

    def user_attr_data_to_pandas_df(self):
        """
        TODO: method documentation here
        """
        import pandas as pd

        pd_df_list = []
        for user_id in self.user_dict:
            pd_df_list.append(
                pd.DataFrame(
                    {k: self.user_dict[user_id][k] for k in self.potential_user_attr},
                    index=[user_id],
                )
            )
        pd_df = pd.concat(pd_df_list, axis=0)
        pd_df.index.name = "user_ID"
        return pd_df

    def item_attr_data_to_pandas_df(self):
        """
        TODO: method documentation here
        """
        import pandas as pd

        pd_df_list = []
        for item_id in self.item_dict:
            pd_df_list.append(pd.DataFrame(self.item_dict[item_id], index=[item_id]))
        pd_df = pd.concat(pd_df_list, axis=0)
        pd_df.index.name = "item_ID"
        return pd_df

    def generate_tutorial(self):
        """
        method documentation TODO
        """
        from pprint import pprint, pformat
        import numpy as np
        import copy

        open_curly = "{"
        closed_curly = "}"
        print(
            f"""
## PLEASE NOTE: this tutorial method was coded in a big hurry ..
##              ..but I thought that it was too important to leave out
##              ..future iterations of it will be increasingly polished  
\n
>> sim_obj = recsys_data_simulator(
        n_users={self.n_users},
        n_items={self.n_items},
        n_user_types={self.n_user_types},
        n_item_attr_pref_mutations_per_user={self.n_item_attr_pref_mutations_per_user},
        n_additional_context_modifiers_per_user={self.n_additional_context_modifiers_per_user},
        potential_item_attr={pformat(self.potential_item_attr, indent=16)},
        potential_user_attr={pformat(self.potential_user_attr,indent=16)},
        potential_context_attr={pformat(self.potential_context_attr, indent=16)},
        rating_range={self.rating_range},
        rating_trunc_norm_std_dev={self.rating_trunc_norm_std_dev},
        n_context_effects={self.n_context_effects},
        context_effect_abs_size={self.context_effect_abs_size},
)

## the simulated items are stored in .item_dict
>> from pprint import pprint
>> pprint(sim_obj.item_dict)
        """
        )

        pprint(self.item_dict)

        print(
            """
## the simulated user types are stored in .user_types_dict
>> pprint(sim_obj.user_types_dict, depth=3)
        """
        )

        pprint(self.user_types_dict, depth=3)

        print(
            """
## the actual simulated user population is stored in .user_dict
## (you can see for each user which base type they were created from)
>> pprint(sim_obj.user_dict, depth=2)
        """
        )

        pprint(self.user_dict, depth=2)

        print(
            """
## let us consider the first simulated user in the population:
>> pprint(sim_obj.user_dict[0])
        """
        )
        pprint(self.user_dict[0])

        print(
            f"""
## this first user was created by mutating user_type={self.user_dict[0]["user_type_ID"]}:
>> pprint(sim_obj.user_dict[0]["user_type_ID"])
        """
        )
        pprint(self.user_dict[0]["user_type_ID"])

        print(
            f"""
## a user's attributes directly match the attributes of the user_type from which they were created:
>> pprint(
>>          {open_curly}
>>              k: sim_obj.user_types_dict[{self.user_dict[0]["user_type_ID"]}][k]
>>              for k in sim_obj.potential_user_attr
>>          {closed_curly}
>>      )
        """
        )
        pprint(
            {
                k: self.user_types_dict[self.user_dict[0]["user_type_ID"]][k]
                for k in self.potential_user_attr
            }
        )

        print(
            f"""
>> pprint(
>>          {open_curly}
>>              k: sim_obj.user_dict[0][k]
>>              for k in sim_obj.potential_user_attr
>>          {closed_curly}
>>      )
        """
        )
        pprint({k: self.user_dict[0][k] for k in self.potential_user_attr})

        print(
            f"""
## Here are the item attribute preferences of user_type={self.user_dict[0]["user_type_ID"]}:
## (a user's preference for each item attribute is described by an integer score from [0,10])
>> pprint(sim_obj.user_types_dict[{self.user_dict[0]["user_type_ID"]}]["relative_item_attr_preferences"]
        """
        )
        pprint(
            self.user_types_dict[self.user_dict[0]["user_type_ID"]][
                "relative_item_attr_preferences"
            ]
        )

        print(
            f"""
## Here, for example, are the item attribute preferences of the first simulated user:
## (the same as user_type={self.user_dict[0]["user_type_ID"]}, but with [n_item_attr_pref_mutations_per_user]={self.n_item_attr_pref_mutations_per_user} mutations)
"""
        )
        print(
            f"""
>> pprint(sim_obj.user_dict[0]["mutations_log"]["item_attribute_preferences"])
        """
        )
        pprint(self.user_dict[0]["mutations_log"]["item_attribute_preferences"])

        print(
            """
>> pprint(sim_obj.user_dict[0]["relative_item_attr_preferences"])
        """
        )
        pprint(self.user_dict[0]["relative_item_attr_preferences"])

        print(
            f"""
## Similarly, here are the context modifiers for user_type={self.user_dict[0]["user_type_ID"]}:
##      ..there are {self.n_context_effects} modifiers per unique context dimension (this is specified using [n_context_effects])
##      ..each modifier shifts the users preference for a particular item attribute by the amount shown (the effect strength is specified using [context_effect_abs_size])
>> pprint(sim_obj.user_types_dict[{self.user_dict[0]["user_type_ID"]}]["context_modifiers"]
        """
        )
        pprint(
            self.user_types_dict[self.user_dict[0]["user_type_ID"]]["context_modifiers"]
        )

        print(
            f"""
## ..and here are the context modifiers for the first simulated user:
## (the same as user_type={self.user_dict[0]["user_type_ID"]}, but with [n_additional_context_modifiers_per_user]={self.n_additional_context_modifiers_per_user} additional modifiers)
>> pprint(sim_obj.user_dict[0]["context_modifiers"])
        """
        )
        pprint(self.user_dict[0]["context_modifiers"])

        random_recommend_context = self.generate_random_context()

        print(
            f"""
## you can calculate the preference for a particular item for a particular user using the .calc_user_preference_for_item() method
>> sim_obj.calc_user_preference_for_item(
            user_id = 1, 
            item_id = 2, 
            recommend_context = {f'{pformat(random_recommend_context,indent=4)}'}
        )
        """
        )
        pprint(
            self.calc_user_preference_for_item(
                user_id=1, item_id=2, recommend_context=random_recommend_context
            )
        )
        print(
            f"""
## here is a breakdown of the calculation, ignoring context:
        
        base affinity score of user_id=1 for item_id=2:     0
        
        item attributes of item_id=2:
>>  pprint( sim_obj.item_dict[2] )
        """
        )
        pprint(self.item_dict[2])
        print(
            f"""
        relevant context modifiers of user_id=1:
        """
        )
        for context_categ in random_recommend_context:
            print(f"{context_categ}='{random_recommend_context[context_categ]}'")
            for mod in self.user_dict[1]["context_modifiers"][context_categ][
                random_recommend_context[context_categ]
            ]:
                print(f"\t{mod}")
        print(
            """
        relevant item attribute preferences of user_id=1:
        """
        )
        context_adj_user_item_attr_prefs = copy.deepcopy(
            self.user_dict[1]["relative_item_attr_preferences"]
        )
        for context_categ in random_recommend_context:
            context_categ_val = random_recommend_context[context_categ]
            for context_mod in self.user_dict[1]["context_modifiers"][context_categ][
                context_categ_val
            ]:
                mod_val = (
                    context_adj_user_item_attr_prefs[context_mod[0]][context_mod[1]]
                    + context_mod[2]
                )
                if (
                    mod_val <= self.rating_range["max"]
                    and mod_val >= self.rating_range["min"]
                ):
                    context_adj_user_item_attr_prefs[context_mod[0]][
                        context_mod[1]
                    ] = mod_val

        store_attr_vals = []
        context_adj_store_attr_vals = []
        for attr in self.item_dict[2]:
            attr_val = self.item_dict[2][attr]
            user_attr_val_affinity = self.user_dict[1][
                "relative_item_attr_preferences"
            ][attr][attr_val]
            store_attr_vals.append(user_attr_val_affinity)
            context_adj_user_attr_val_affinity = context_adj_user_item_attr_prefs[attr][
                attr_val
            ]
            context_adj_store_attr_vals.append(context_adj_user_attr_val_affinity)
            if user_attr_val_affinity == context_adj_user_attr_val_affinity:
                print(f"{attr}={attr_val}: {user_attr_val_affinity}")
            else:
                print(
                    f"{attr}={attr_val}: {user_attr_val_affinity} (context adjusted: {context_adj_user_attr_val_affinity})"
                )
        print(
            f"total (unscaled) affinity score of user_id=1 for item_id=2:     {' + '.join([str(x) for x in store_attr_vals])} = {sum(store_attr_vals)}"
        )
        print(
            f"total context-adjusted (unscaled) affinity score of user_id=1 for item_id=2:     {' + '.join([str(x) for x in context_adj_store_attr_vals])} = {sum(context_adj_store_attr_vals)}"
        )

        print(
            f"""unscaled rating is scaled to lie in the user-specified possible rating range [{self.rating_range['min']},{self.rating_range['max']}]
            (rating range is specified using [rating_range])
            total (scaled) affinity score of user_id=1 for item_id=2:   {self.rating_range['min']} + {sum(store_attr_vals)} * ({self.rating_range['max']}-{self.rating_range['min']}) / {self.max_achievable_preference_score} = {self.rating_range['min']+sum(store_attr_vals) * (self.rating_range['max']-self.rating_range['min']) / self.max_achievable_preference_score} 
            total context-adjusted (scaled) affinity score of user_id=1 for item_id=2:   {self.rating_range['min']} + {sum(context_adj_store_attr_vals)} * ({self.rating_range['max']}-{self.rating_range['min']}) / {self.max_achievable_preference_score} = {self.rating_range['min']+sum(context_adj_store_attr_vals) * (self.rating_range['max']-self.rating_range['min']) / self.max_achievable_preference_score} 
                ..highest rating is achieved with an unscaled score of {self.max_achievable_preference_score} (preference of 10 for each item attribute)
                ..lowest rating is achieved with an unscaled score of 0 (preference of 0 for each item attribute)
        """
        )


if __name__ == "__main__":
    ## run illustrative example ##
    sim_obj = recsys_data_simulator(
        n_users=5,
        n_items=3,
        n_user_types=3,
        n_item_attr_pref_mutations_per_user=2,
        n_additional_context_modifiers_per_user=1,
        potential_item_attr={
            "colour": [
                "black",
                "white",
            ],
            "size": ["small", "medium", "large"],
            "material": [
                "metal",
                "wood",
                "cotton",
                "plastic",
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
        },
        rating_range={
            "min": 1,
            "max": 10,
        },
        rating_trunc_norm_std_dev=0.2,
        n_context_effects=2,
        context_effect_abs_size=2,
    )
    print(sim_obj.generate_tutorial())

    # sim_obj = recsys_data_simulator(
    #     n_users=10,
    #     n_items=5,
    #     n_user_types=3,
    #     n_item_attr_pref_mutations_per_user=2,
    #     n_context_pref_mutations_per_user=2,
    #     potential_item_attr={
    #         "colour": [
    #             "red",
    #             "green",
    #             "blue",
    #             "black",
    #         ],
    #         "size": ["small", "medium", "large"],
    #         "material": [
    #             "metal",
    #             "wood",
    #             "cotton",
    #             "plastic",
    #         ],
    #     },
    #     potential_user_attr={
    #         "location": ["cape town", "london", "dubai", "new york", "rotterdam"],
    #         "age": ["infant", "teenager", "youth", "middle_aged", "old"],
    #     },
    #     potential_context_attr={
    #         "time_of_day": ["morning", "afternoon", "night"],
    #         "day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    #         "social_context": [
    #             "public_space",
    #             "public_transport",
    #             "private_space",
    #             "private_transport",
    #         ],
    #         "user_group_recommendation": [
    #             "user_alone",
    #             "small_user_group",
    #             "large_user_group",
    #         ],
    #     },
    #     rating_range={"min": 1, "max": 5},
    #     rating_trunc_norm_std_dev=0.2,
    #     n_context_effects=3,
    #     context_effect_abs_size=2,
    # )

    # pprint(sim_obj.item_dict[1])  # attributes of item 1
    # pprint(sim_obj.user_dict[3])  # attributes of user 3
    # user_type_id = sim_obj.user_dict[3][
    #     # base user "stereotype" from which user 3 was created
    #     # i.e. user 3 the same as this type, with a few little mutations
    #     "user_type_ID"
    # ]
    # pprint(sim_obj.user_types_dict[user_type_id])
    # sim_obj.calc_user_preference_for_item(
    #     user_id=3,
    #     item_id=1,
    #     recommend_context={
    #         "time_of_day": "morning",
    #         "day_of_week": "tuesday",
    #         "social_context": "public_transport",
    #         "user_group_recommendation": "user_alone",
    #     },
    # )
    # sim_obj.expose_user_to_item(
    #     user_id=3,
    #     item_id=1,
    #     recommend_context={
    #         "time_of_day": "morning",
    #         "day_of_week": "tuesday",
    #         "social_context": "public_transport",
    #         "user_group_recommendation": "user_alone",
    #     },
    #     ignore_context=True,
    #     log_interaction=False,
    # )
    # sim_obj.expose_user_to_item(
    #     user_id=3,
    #     item_id=0,
    #     recommend_context={
    #         "time_of_day": "morning",
    #         "day_of_week": "tuesday",
    #         "social_context": "public_transport",
    #         "user_group_recommendation": "user_alone",
    #     },
    #     ignore_context=False,
    #     log_interaction=False,
    # )
    # sim_obj.expose_user_to_item(
    #     user_id=3,
    #     item_id=0,
    #     recommend_context={
    #         "time_of_day": "morning",
    #         "day_of_week": "tuesday",
    #         "social_context": "public_transport",
    #         "user_group_recommendation": "user_alone",
    #     },
    #     ignore_context=False,
    #     log_interaction=True,
    # )
    # pprint(sim_obj.user_dict[3])
    # sim_obj.user_item_exposure_history_to_pandas_df()

    # pprint(sim_obj.item_dict)
    # sim_obj.item_attr_data_to_pandas_df()
