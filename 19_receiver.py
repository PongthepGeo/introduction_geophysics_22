import sys
sys.path.append('./Libs') 
import seismic as S
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
#-----------------------------------------------------------------------------------------#

'''
step 1: importing image with fixed model size at 401x401 pixels
'''

image = imageio.imread('data/fault.png') / 255
image = image[:, :, 0]
image = S.convert2velocity(image, 1)
# S.plot_velocity_custom(image, 'Geological Structure')

'''
step 2: record wave propagation at each geophone (receiver) location.
'''

S.receiver_pipeline(space_order = 6,
                    time_order = 2,
                    image = image,
                    simulation_time_start = 0.,
					# simulation_time_stop = 320., # timestep 17
					# simulation_time_stop = 450., # timestep 27
					# simulation_time_stop = 580., # timestep 37
					simulation_time_stop = 1500.,
					source_frequency = 0.04, # unit MHz
                    source_depth = 20 # unit meter
                    )