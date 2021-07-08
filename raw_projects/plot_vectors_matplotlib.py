import matplotlib.pyplot as plt              

fig, ax = plt.subplots(figsize=(8, 8))     # make the plot nice and big

x = [1,2,3,4,5]
y = [5,4,3,2,1]
labels = ['the', 'quick','brown','fox','jumped']

origin = [0] # define origin point
plt.quiver( [0]*len(x), [0]*len(y), x, y, scale=10)


#for i, txt in enumerate(labels):
#    ax.annotate(txt, (x[i], y[i]))

plt.show()