import sys
sys.path.append('./Libs') 
import function as f
#-----------------------------------------------------------------------------------------#
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
#-----------------------------------------------------------------------------------------#

seismic = imageio.imread('image_out/seismic.png') / 255
seismic = seismic[:, :, 0]
plt.imshow(seismic)
# plt.show()

avg_freq = 0.; count = 0
beg_X = 150; end_X = 385
beg_Y = 180; end_Y = 340
for i in range (beg_X, end_Y):
	freq = f.compute_frequency(seismic[beg_Y:end_Y, i], 100)
	avg_freq = avg_freq + freq
	count += 1
print(avg_freq/count)