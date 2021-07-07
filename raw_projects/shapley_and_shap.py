
import itertools
import re
import pandas as pd
import numpy as np

items = ['A','B','C','D']

all_item_subsets = ( 
    items +
    [ i[0]+i[1] for i in list( itertools.combinations(items, 2) ) ] +
    [ i[0]+i[1]+i[2] for i in list( itertools.combinations(items, 3) ) ] +
    ['ABCD']
)

game1_power_ref = {'A':8, 'B':2, 'C':3, 'D':6}
game2_power_ref = {'A':5, 'B':2, 'C':4, 'D':3}

def calc_subset_power_game1( item_subset, game_choice ):
    if game_choice == 1:
        return ( np.select( ['A' in item_subset], [game1_power_ref['A']], [0] )[0] +
                 np.select( ['B' in item_subset], [game1_power_ref['B']], [0] )[0] +
                 np.select( ['C' in item_subset], [game1_power_ref['C']], [0] )[0] +
                 np.select( ['D' in item_subset], [game1_power_ref['D']], [0] )[0]
               )
    elif game_choice == 2:
        return ( np.select( ['A' in item_subset], [game2_power_ref['A']], [0] )[0] +
                 np.select( ['B' in item_subset], [game2_power_ref['B']], [0] )[0] +
                 np.select( ['C' in item_subset], [game2_power_ref['C']], [0] )[0] +
                 np.select( ['D' in item_subset], [game2_power_ref['D']], [0] )[0]
               )
    else:
        return 'ERROR' 

total_power_each_subset_game1 = [ calc_subset_power_game1(item_subset=i,game_choice=1) for i in all_item_subsets ]
total_power_each_subset_game2 = [ calc_subset_power_game1(item_subset=i,game_choice=2) for i in all_item_subsets ]

def calc_output( total_power, game_choice ):
    if game_choice==1 and total_power >= 18:
        return 200
    elif game_choice==1 and total_power >= 10:
        return 100
    elif game_choice==1 and total_power >= 8:
        return 50
    elif game_choice==1 and total_power < 8:
        return 0
    elif game_choice==2 and total_power >= 14:
        return 200
    elif game_choice==2 and total_power >= 6:
        return 100
    elif game_choice==2 and total_power >= 5:
        return 50
    else:
        return 0 

total_output_each_subset_game1 = [ calc_output(x, game_choice=1) for x in total_power_each_subset_game1 ]
total_output_each_subset_game2 = [ calc_output(x, game_choice=2) for x in total_power_each_subset_game2 ]
total_output_each_subset_game3 = [ x+y for x,y in zip(total_output_each_subset_game1,total_output_each_subset_game2) ] 

pd.DataFrame( 
                { 
                    'subset':all_item_subsets,
                    'GAME1_total_power':total_power_each_subset_game1,
                    'GAME2_total_power':total_power_each_subset_game2,
                    'GAME1_total_output':total_output_each_subset_game1,
                    'GAME2_total_output':total_output_each_subset_game2,
                    'GAME3_total_output':total_output_each_subset_game3
                }   
            )            
# old code:

cost_ref = {'0':0,'A':8,'B':11,'C':13,'D':18}                   

def get_subset_cost( item_subset ):
    most_expensive_item_in_subset = item_subset[ len(item_subset)-1 ]
    return  cost_ref[ most_expensive_item_in_subset ]

def get_marginal_item_cost( 
                            item_subset,         # e.g. 'ABD' 
                            item_get_value_for   # e.g. 'B'
                          ):
    item_subset_with_item_removed = re.sub( item_get_value_for, '', item_subset )
    if item_subset_with_item_removed == item_subset: 
        return ''
    elif len(item_subset_with_item_removed) == 0:
        return get_subset_cost(item_get_value_for)
    else:
        return get_subset_cost(item_subset) - get_subset_cost(item_subset_with_item_removed) 

shapley_calc = pd.DataFrame( 
                {
                'subset':all_item_subsets,
                'subset_size':[ len(x) for x in all_item_subsets ],
                'marg_cost_0':[ get_marginal_item_cost(item_subset=x, item_get_value_for='0') for x in all_item_subsets ],
                'marg_cost_A':[ get_marginal_item_cost(item_subset=x, item_get_value_for='A') for x in all_item_subsets ],
                'marg_cost_B':[ get_marginal_item_cost(item_subset=x, item_get_value_for='B') for x in all_item_subsets ],
                'marg_cost_C':[ get_marginal_item_cost(item_subset=x, item_get_value_for='C') for x in all_item_subsets ],
                'marg_cost_D':[ get_marginal_item_cost(item_subset=x, item_get_value_for='D') for x in all_item_subsets ]
                }
            )
shapley_calc            