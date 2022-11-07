import sys
sys.path.append('./Libs') 
import function as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
#-----------------------------------------------------------------------------------------#

'''
step 1: predefine signal length
'''

dx = 0.001
L = np.pi
x = L * np.arange(-1+dx,1+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))

'''
step 2: synthetic signal, creating from a combination of sine and cosince functions
'''

amplitude = 2  # no unit
frequency = 5 # Hz
time = np.linspace(0, np.pi/32, 2000) # second
phase = np.pi/2 # where to start (radians)
y_5 = F.sin_cos(amplitude, 5, time, phase, 'cos')
y_15 = F.sin_cos(amplitude, 15, time, phase, 'sin')
y_45 = F.sin_cos(amplitude, 45, time, phase, 'cos')
f = y_5 + y_15 + y_45

'''
step 3: plot result
'''

fig, ax = plt.subplots(figsize=(20, 15))
# NOTE plot synthetic signal
ax.plot(x, f, '-', color='b', linewidth=5)
# NOTE assign colors
cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)
# NOTE coefficients A0
A0 = np.sum(f * np.ones_like(x)) * dx
fFS = A0/2
# NOTE compute coefficients A and B
number_of_generating_function = 5
A = np.zeros(number_of_generating_function)
B = np.zeros(number_of_generating_function)
for k in range(number_of_generating_function):
	A[k] = np.sum(f * np.cos(np.pi*(k+1)*x/L)) * dx # Inner product
	B[k] = np.sum(f * np.sin(np.pi*(k+1)*x/L)) * dx
	fFS = fFS + A[k]*np.cos((k+1)*np.pi*x/L) + B[k]*np.sin((k+1)*np.pi*x/L)
	ax.plot(x, fFS, '--')
# plt.savefig('image_out/' + 'fourier_series' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()
