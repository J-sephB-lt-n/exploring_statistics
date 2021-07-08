# https://stackoverflow.com/questions/46448661/matplotlib-how-to-plot-the-difference-of-two-histograms/46449178
# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')

# define first x vector:
x_plot1 = np.random.randint(1,30,1000)

# define second x vector:
x_plot2 = np.random.randint(1,30,1000)

# set width of bins:
binwidth = 5
binrange = range(0,40,binwidth)

# get binning data for each of the individual histograms:
x1_histogram_binning = np.histogram(x_plot1, bins=binrange)
x2_histogram_binning = np.histogram(x_plot2, bins=binrange)

# calculate the difference between the 2 densities:
x1_densities = x1_histogram_binning[0]/sum(x1_histogram_binning[0])
x2_densities = x2_histogram_binning[0]/sum(x2_histogram_binning[0])
density_diff = []
for i in range(len(x1_densities)):
    difference = x1_densities[i] - x2_densities[i]
    density_diff.append(difference)

# set up the stacked plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,6), sharex=True )    # stack histograms 3 rows by 2 columns
fig.suptitle('Grand Title')

# plot the histograms and difference plot:
ax1.hist( x_plot1, density=False, bins=binrange )
ax2.hist( x_plot2, density=False, bins=binrange )
ax3.bar( x = x1_histogram_binning[1] + binwidth/2,
         height = density_diff + [0],
         color = 'red',
         width = binwidth/1.5
       )

# label the plots:       
ax1.set_title('Distribution of something 1')
ax2.set_title('Distribution of something 2')
ax3.set_title('Difference Between Densities')
ax1.set_ylabel('Number of Orders')
ax2.set_ylabel('Number of Orders')
ax3.set_ylabel('Density Difference')


plt.show()
