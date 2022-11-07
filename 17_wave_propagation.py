import sys
sys.path.append('./Libs') 
import seismic as S
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt 
#-----------------------------------------------------------------------------------------#
from examples.seismic import Receiver, RickerSource, Model, plot_velocity, TimeAxis
from devito import TimeFunction, Eq, solve, Operator, ConditionalDimension
from scipy import ndimage
#-----------------------------------------------------------------------------------------#

'''
step 1: create velocity model
'''

# NOTE define model size
nx = 401; nz = 401; nb = 100
shape = (nx, nz)
spacing = (6., 6.) # grid spacing
origin = (0., 0.)
# NOTE velocity
v = np.empty(shape, dtype=np.float32)
v[:, int(nx/2):] = 2.1 # top
v[:, :int(nx/2)] = 3.5 # bottom
# v[150:170, 50:70] = 4.6. # box
# NOTE plot velocity, we need to rotate 90 for visualing purpose. Devito uses different coordinate from plt.imshow.
# S.plot_velocity(v, 'Two Layers')
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=10, nbl=100, bcs='damp')

'''
step 2: create layout for source and geophone acquisition
'''

t0 = 0.  # Simulation starts a t=0
tn = 1400.  # Simulation lasts tn milliseconds
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
nt = time_range.num  # number of time steps
# NOTE source position
f0 = 0.02  # Source peak frequency (0.020 kHz)
src = RickerSource(
	name='src',
	grid=model.grid,
	f0=f0,
	time_range=time_range)  
# source layout
# src.coordinates.data[0, :] = np.array(model.domain_size) * .5 
src.coordinates.data[0, :] = (np.array(model.domain_size) - (nb*2*6)) * .5 
src.coordinates.data[0, -1] = 50 - nb*6  
# NOTE reciever positions
rec = Receiver(
	name='rec',
	grid=model.grid,
	npoint=101,
	time_range=time_range)  # new
rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.  # depth is 20
depth = rec.coordinates.data[:, 1]  
# NOTE plot velocity model and acquisition
plot_velocity(model, source=src.coordinates.data, receiver=rec.coordinates.data[::4, :])

'''
step 3: locate point source and compute stencil
'''

# NOTE locate source
vnx = nx+nb*2 # Used for reshaping model
vnz = nz+nb*2
nsnaps = 103 
# nsnaps = 203 
factor = round(nt / nsnaps) 
print(f"factor is {factor}")
# NOTE define finite difference (mathematical method to solve wave equation)
time_subsampled = ConditionalDimension('t_sub', parent=model.grid.time_dim, factor=factor)
usave = TimeFunction(name='usave', grid=model.grid, time_order=2, space_order=4, save=nsnaps, time_dim=time_subsampled)
print(time_subsampled)
# NOTE inject point source
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=4)
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
stencil = Eq(u.forward, solve(pde, u.forward))
src_term = src.inject(
	field=u.forward,
	expr=src * dt**2 / model.m,
	offset=model.nbl)
rec_term = rec.interpolate(expr=u, offset=model.nbl)

'''
step 4: solve wave equation
'''

op1 = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)  # usual operator
op2 = Operator([stencil] + src_term + [Eq(usave, u)] + rec_term, subs=model.spacing_map)  # operator with snapshots
op1(time=nt - 2, dt=model.critical_dt)  # run only for comparison
u.data.fill(0.)
op2(time=nt - 2, dt=model.critical_dt)

'''
step 5: save snapshot as .npy
'''

print('Saving snaps file')
print('Dimensions: nz = {:d}, nx = {:d}'.format(nz + 2 * nb, nx + 2 * nb))
filename = '../snapshots/snaps2.bin'
usave.data.tofile(filename)
fobj = open(filename, 'rb')
snaps = np.fromfile(fobj, dtype=np.float32)
snaps = np.reshape(snaps, (nsnaps, vnx, vnz))
fobj.close()

'''
step 6: load computed snapshots from .npy files and plot wave propagation in each time step.
'''

plt.rcParams['figure.figsize'] = (20, 20)  # Increases figure size
imcnt = 1 # Image counter for plotting
plot_num = 5 # Number of images to plot
# plot_num = 10 # Number of images to plot
snap = '../snapshots/timestep_'
for i in range(0, plot_num):
	imcnt = imcnt + 1
	ind = i * int(nsnaps/plot_num)
	np.save(snap + str(imcnt), np.transpose(snaps[ind, :, :]))

imcnt = 1 # Image counter for plotting
for i in range(0, plot_num):
	imcnt = imcnt + 1
	ind = i * int(nsnaps/plot_num)
	dummy = np.load(snap + str(imcnt) + '.npy')
	plt.imshow(dummy, cmap='gray')
	plt.title('Time Step: ' + str(imcnt))
	# plt.xlim(40, 200)
	# plt.ylim(200, 0)
	# save_file = ('timestep_' + str(imcnt))
	# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()