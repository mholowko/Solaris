# create plot dictionaries:
# keys are the group name in master file
# values:
#   name_dict: name of group show in plot
#   color_dict: color of group 
#   loc_dictt: location of group

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle

name_dict = {}
color_dict = {}
loc_dict = {}

name_dict['consensus'] = 'Consensus'
name_dict['reference'] = 'Reference'
name_dict['bps_core'] = 'BPS-C'
name_dict['bps_noncore'] = 'BPS-NC', 
name_dict['uni random'] = 'UNI'
name_dict['prob random'] = 'PPM' 
name_dict['bandit'] = 'Bandit-0'
name_dict['bandit2'] = 'Bandit-1'
name_dict['bandit3'] = 'Bandit-2'
name_dict['bandit4'] = 'Bandit-3'


loc_dict['consensus'] = 0
loc_dict['reference'] = 0
loc_dict['bps_core'] = 1
loc_dict['bps_noncore'] = 2 
loc_dict['uni random'] = 3
loc_dict['prob random'] = 4
loc_dict['bandit'] = 5
loc_dict['bandit2'] = 6
loc_dict['bandit3'] = 7
loc_dict['bandit4'] = 8

# loc_dict['consensus'] = 0
# loc_dict['reference'] = 1
# loc_dict['bps_core'] = 2
# loc_dict['bps_noncore'] = 3 
# loc_dict['uni random'] = 4
# loc_dict['prob random'] = 5
# loc_dict['bandit'] = 6
# loc_dict['bandit2'] = 7
# loc_dict['bandit3'] = 8
# loc_dict['bandit4'] = 9


# Follow tutorial https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
viridis = cm.get_cmap('viridis', 5)
tab10 = cm.get_cmap('tab10', 10)

color_dict['consensus'] = tab10.colors[1]
color_dict['reference'] = tab10.colors[3]
color_dict['bps_core'] = tab10.colors[5]
color_dict['bps_noncore'] = tab10.colors[6]
color_dict['uni random'] = tab10.colors[7]
color_dict['prob random'] = tab10.colors[8]
color_dict['bandit'] = viridis.colors[0]
color_dict['bandit2'] = viridis.colors[1]
color_dict['bandit3'] = viridis.colors[2]
color_dict['bandit4'] = viridis.colors[3]

# for name in name_dict.keys():
#     plt.scatter(loc_dict[name], loc_dict[name], color = color_dict[name])
# # plt.xticks(list(loc_dict.values()), list(name_dict.values()))
# # plt.title('viridis (0-3) + tab 10')
# plt.show()

# for i in range(10):
#     plt.scatter(i, i, color = tab10.colors[i])
# plt.show()

plot_dict={}
plot_dict['name'] = name_dict
plot_dict['loc'] = loc_dict
plot_dict['color'] = color_dict

# np.save('plot_dict.npy', plot_dict)