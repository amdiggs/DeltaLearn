import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.style.use('seaborn-deep')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['mediumblue', 'crimson','darkgreen', 'darkorange', 'black', 'darkorchid','cyan'])
plt.rcParams['figure.figsize'] = [10,8]
plt.rcParams['axes.linewidth'] = 1.7
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['font.family'] =  'sans-serif'
#plt.rcParams['font.cursive'] = ['Calibri']
plt.rcParams['patch.linewidth'] = 2.5

