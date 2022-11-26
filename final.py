import sys
sys.path.append('./Libs') 
import seismic as S
import function as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
import matplotlib.colors as colors
#-----------------------------------------------------------------------------------------#

'''
step 1: importing image with fixed model size at 401x401 pixels
'''

image = imageio.imread('data/model.png') / 255
image = image[:, :, 0]
image = S.compute_kmeans(image, 20)
image = S.convert2velocity(image, (1, 2))
# image = S.convert2velocity(image, (1, 1))
# S.plot_velocity_custom(image, 'Geological Structure')

# after_kmeans = S.compute_kmeans(image, 4)
# after_kmeans = S.compute_kmeans(image, 50)

# k_dummy = np.unique(after_kmeans)
# for i in range (200, image.shape[0]):
# 	for ii in range (0, image.shape[1]):
# 		if after_kmeans[i, ii] == k_dummy[-1]:
# 			after_kmeans[i, ii] = k_dummy[2]

# lithocolors  = ['#FCF3CF', # yellow 
#                 '#704807', # brown
#                 '#7FB3D5'] # blue
# cmap = colors.ListedColormap(lithocolors)

# plt.imshow(after_kmeans, cmap=cmap)
# plt.imshow(image, cmap='terrain')
# plt.colorbar()
# plt.savefig('data/kmeans.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()

'''
step 2: generate wave propagation.
'''

# S.wave_propagation_pipeline(space_order = 6,
#                             time_order = 2,
#                             image = image,
#                             simulation_time_start = 0.,
# 							simulation_time_stop = 1800.,
# 							source_frequency = 0.04, # unit MHz
#                             source_depth = 200, # unit meter
#                             number_of_snapshot = 103,
#                             number_of_save_snapshot = 7)

'''
step 2: record wave propagation at each geophone (receiver) location.
'''

# S.receiver_pipeline(space_order = 6,
#                     time_order = 2,
#                     image = image,
#                     simulation_time_start = 0.,
# 					# simulation_time_stop = 320., # timestep 17
# 					# simulation_time_stop = 450., # timestep 27
# 					# simulation_time_stop = 580., # timestep 37
# 					simulation_time_stop = 2500.,
# 					source_frequency = 0.04, # unit MHz
#                     source_depth = 20 # unit meter
#                     )

seismic = F.reflectivity(image, 15)
# seismic = F.reflectivity(after_kmeans, 15)
# seismic = F.ref_lap_but(image, 25)
max_num, min_num = F.clip(seismic, 99)
plt.imshow(seismic, vmin=min_num, vmax=max_num, cmap='seismic')
# plt.savefig('data/seismic.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()