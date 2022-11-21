import sys
sys.path.append('./Libs') 
import seismic as S
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
#-----------------------------------------------------------------------------------------#

'''
step 1: create simple 3 layers
'''

multiple_model = np.zeros(shape=(401, 401), dtype=np.float32)
multiple_model[:, :200] = 2.6 # top
multiple_model[:, 200:250] = 5.1 # middle
multiple_model[:, 250:] = 2.3 # bottom

# S.plot_velocity_custom(np.transpose(multiple_model), 'm')

'''
step 2: generate wave propagation.
'''

# S.wave_propagation_pipeline(space_order = 6,
#                             time_order = 2,
#                             image = np.transpose(multiple_model),
#                             simulation_time_start = 0.,
# 							simulation_time_stop = 1200.,
# 							source_frequency = 0.04, # unit MHz
#                             source_depth = 20, # unit meter
#                             number_of_snapshot = 403,
#                             number_of_save_snapshot = 8)

'''
step 3: receivers
'''

S.receiver_pipeline(space_order = 6,
                    time_order = 2,
                    image = np.transpose(multiple_model),
                    simulation_time_start = 0.,
					simulation_time_stop = 1500.,
					source_frequency = 0.04, # unit MHz
                    source_depth = 20 # unit meter
                    )