import sys
sys.path.append('./Libs') 
import function as f
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

amplitude = 2  # no unit
frequency = 5 # Hz
time = np.linspace(0, np.pi/32, 1000) # second
phase = np.pi/2 # where to start (radians)

f.sin_cos_plot(amplitude, frequency, time, phase, 'cos', 'Frequency: 5')

# fig = plt.figure(figsize=(10, 5))
# y_5 = f.sin_cos(amplitude, 5, time, phase, 'cos')
# y_15 = f.sin_cos(amplitude, 15, time, phase, 'cos')
# y_45 = f.sin_cos(amplitude, 45, time, phase, 'cos')
# y = y_5 + y_15 + y_45
# plt.plot(time, y_5, color='cyan', linestyle='--', linewidth=2)
# plt.plot(time, y_15, color='green', linestyle='--', linewidth=2)
# plt.plot(time, y_45, color='red', linestyle='--', linewidth=2)
# plt.plot(time, y, color='black', linestyle='-', linewidth=5)
# plt.xticks(fontsize=16); plt.yticks(fontsize=16)
# plt.xlabel(r'radian ($\pi$)', fontsize=18)
# plt.ylabel('amplitude', fontsize=18)
# # plt.title('Summation', fontsize=18, fontweight='bold')
# # plt.savefig('image_out/' + 'Summation' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()