import sys
sys.path.append('./Libs') 
import function as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

time = np.linspace(0, np.pi/32, 2000) # second
phase = np.pi/2 # where to start (radians)
amplitudes = [2, 4, 1]
frequencies = [15, 25, 45]
functions = ['cos', 'sin', 'cos']
wave = 0.
for index, i in enumerate(amplitudes):
	dummy = F.sin_cos(amplitudes[index], frequencies[index], time, phase, functions[index])
	wave = wave + dummy

c = 1500
plt.figure(figsize=(20, 15))
plt.rcParams.update({'font.size': 22})

# plt.plot(wave, '-', color='blue', linewidth=5)
# plt.plot(np.gradient(wave), '-', color='orange', linewidth=1)
# plt.plot(np.gradient(np.gradient(wave)), '-', color='green', linewidth=1)  

# plt.plot(wave*c**2, '--', color='black', linewidth=5)
# plt.plot(np.gradient(wave)*c**2, '--', color='salmon', linewidth=1)
plt.plot(np.gradient(np.gradient(wave))*c**2, '-', color='pink', linewidth=1) # 

# plt.title('Wave Propagation in Space', fontweight='bold')
plt.title('Wave Propagation in Space 2$^{st}$ derivative', fontweight='bold')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.savefig('../drawing/image_out/' + 'wavespace_2st' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()
