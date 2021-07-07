# this resource also has STACKED and DODGED bars!!!!
# and placement and tick labelling! 
# https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# 

import matplotlib.pyplot as plt
import seaborn as sns            # for nice styling
sns.set_style("darkgrid")     # set global plot styling

#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
fig, ax = plt.subplots(figsize=(12, 12))

x_axis_entries = ['C', 'C++', 'Java', 'Python', 'PHP']
y_axis_entries = [23,17,35,29,12]
ax.bar( x = x_axis_entries,
        height = y_axis_entries,
        width = 0.7,       # width of bars
        bottom = None,     # controls bottom of bars
        align = 'center'   # where to place the bar relative to the x-axis tick
     )
ax.set_title('custom title')     
ax.set_xlabel('x axis label custom')
ax.set_ylabel('y axis label custom')
plt.show()