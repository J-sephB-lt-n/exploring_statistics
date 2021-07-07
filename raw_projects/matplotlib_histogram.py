import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')

x = kfc_ordervalues['Total'].values

fig, ax = plt.subplots(figsize=(12, 12))
    
plt.hist(x, density=False, bins=range(0,1000,25))  
plt.ylabel('Number of Orders')
plt.xlabel('Total Order Value (R)')
plt.xticks(range(0,1000,100))