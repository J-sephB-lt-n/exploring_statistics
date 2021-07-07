

def anti_join(x, y, on):
   # https://gist.github.com/sainathadapa/eb3303975196d15c73bac5b92d8a210f
    """Return rows in x which are not present in y"""
    ans = pd.merge(left=x, right=y, how='left', indicator=True, on=on)
    ans = ans.loc[ans._merge == 'left_only', :].drop(columns='_merge')
    return ans

import pandas as pd
main_data_frame = pd.DataFrame({'id':[1,2,3,4,5,6],
                                 'x':[19,18,27,16,15,14]}
                              )
ids_to_remove = pd.DataFrame({'id':[2,5,6],'y':[100,200,300]})

print( main_data_frame)
print(ids_to_remove)

ids_removed = pd.merge(  main_data_frame,
                         ids_to_remove,
                         on = 'id',
                        how = 'left'
                       )

print(ids_removed)
ids_removed = ids_removed[ ids_removed['y'].isnull() ]
print(ids_removed)                       