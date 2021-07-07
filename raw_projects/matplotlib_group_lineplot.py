import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_for_plot = pd.DataFrame( {'group':['A','A','A','B','B','B','C','C','C'],
                                   'x':[ 1,  2,  3,  1,  2,  3,  1,  2,  3 ],
                                   'y':[ np.random.randint(1,10) for i in range(9) ] 
                              } )

fig, ax = plt.subplots(figsize=(12, 12))

for group in data_for_plot['group'].drop_duplicates().values:
    plt.plot( data_for_plot.query('group=="'+group+'"').x.values,
              data_for_plot.query('group=="'+group+'"').y.values,
              marker = '',
              linewidth = 1,
              label = group
            )

plt.xticks(rotation=90)    # rotate ticks on x-axis    

plt.legend()
plt.show()

