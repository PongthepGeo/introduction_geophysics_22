import matplotlib.pyplot as plt
import skimage.io
import numpy as np
#-----------------------------------------------------------------------------------------#
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from examples.seismic import Receiver, RickerSource, Model, plot_velocity, TimeAxis, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from devito import TimeFunction, Eq, solve, Operator, ConditionalDimension
from sklearn.cluster import KMeans
#-----------------------------------------------------------------------------------------#

def plot_velocity_custom(model, title):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
	# NOTE convert model size to km by dividing 100
	extent=[0, model.shape[1]/100, model.shape[0]/100, 0] 
	cax = ax.imshow(model, cmap='jet', extent=extent)
	ax.xaxis.set_label_position('bottom') 
	ax.xaxis.tick_bottom()
	ax.set_xlabel('distance (km)')
	ax.set_ylabel('depth (km)')
	plt.title(title)
	cbar = fig.colorbar(cax, orientation='vertical', fraction=0.047, pad=0.01, shrink=1.0)
	cbar.set_label('velocity (km/s)', labelpad=18, rotation=270)
	# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def scaling_velocity(y, lowest_value, highest_value):
	y = (y - y.min()) / (y.max() - y.min())
	return y * (highest_value - lowest_value) + lowest_value

def convert2velocity(img, filter_sigma):
	img = scaling_velocity(img, 2.0, 4.5)
	return gaussian_filter(img, sigma=filter_sigma)

def clip(model, perc):
	(ROWs, COLs) = model.shape
	reshape2D_1D = model.reshape(ROWs*COLs)
	reshape2D_1D = np.sort(reshape2D_1D)
	if perc != 100:
		min_num = reshape2D_1D[ round(ROWs*COLs*(1-perc/100)) ]
		max_num = reshape2D_1D[ round((ROWs*COLs*perc)/100) ]
	elif perc == 100:
		min_num = min(model.flatten())
		max_num = max(model.flatten())
	if min_num > max_num:
		dummy = max_num
		max_num = min_num
		min_num = dummy
	return max_num, min_num 

def plot_processed_shot(data, perc, t0, tn, title, save_file):
	fig = plt.figure(figsize=(10, 10))
	data = scaling_velocity(data, -1, 1)
	max_num, min_num = clip(data, perc)
	extent = [0, data.shape[1]/100, 1e-3*tn, t0]
	plt.imshow(data, cmap='gray', vmin=min_num, vmax=max_num, extent=extent)
	# plt.imshow(data, cmap='gray', vmin=min_num, vmax=max_num)
	plt.title(title)
	plt.xlabel('trace' + r'$\times 10^3$')
	plt.ylabel('time (s)')
	# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show(); plt.clf

def wave_propagation_pipeline(space_order,
                              time_order,
                              image,
                              simulation_time_start,
                              simulation_time_stop,
					          source_frequency,
                              source_depth,
                              number_of_snapshot,
                              number_of_save_snapshot
                             ):

	'''
	step 1: define model size and geometry. For implicity, the model size is fixed by using nx x nx, 401 x 401. In other words, we use image having pixel x pixel, 401 x 401.
	'''

	# nx = 401; nz = 401; nb = 100
	nx = image.shape[1]; nz = image.shape[0]; nb = 100
	shape = (nx, nz)
	spacing = (6., 6.) # grid spacing
	origin = (0., 0.)
	model = Model(vp=np.transpose(image), origin=origin, shape=shape, spacing=spacing, space_order=space_order, nbl=100, bcs='damp')

	'''
	step 2: create layout for source and geophone acquisition
	'''

	dt = model.critical_dt  # Time step from model grid spacing
	time_range = TimeAxis(start=simulation_time_start, stop=simulation_time_stop, step=dt)
	nt = time_range.num  # number of time steps
	# NOTE source position
	src = RickerSource(name='src', grid=model.grid, f0=source_frequency, time_range=time_range)  
	# NOTE source layout
	src.coordinates.data[0, :] = (np.array(model.domain_size) - (nb*2*6)) * .5 
	src.coordinates.data[0, -1] = 50 - nb*6 # source depth 
	# NOTE reciever positions
	rec = Receiver(name='rec', grid=model.grid, npoint=101, time_range=time_range)
	rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
	rec.coordinates.data[:, 1] = source_depth  # receiver depth
	depth = rec.coordinates.data[:, 1]  
	# NOTE plot velocity model and acquisition
	# plot_velocity(model, source=src.coordinates.data, receiver=rec.coordinates.data[::4, :])

	'''
	step 3: locate point source and compute stencil
	'''

	# NOTE locate source
	vnx = nx+nb*2 # Used for reshaping model
	vnz = nz+nb*2
	factor = round(nt / number_of_snapshot) 
	print(f"factor is {factor}")
	# NOTE define finite difference (mathematical method to solve wave equation)
	time_subsampled = ConditionalDimension('t_sub', parent=model.grid.time_dim, factor=factor)
	usave = TimeFunction(name='usave', grid=model.grid, time_order=time_order, space_order=space_order, save=number_of_snapshot, time_dim=time_subsampled)
	print(time_subsampled)
	# NOTE inject point source
	u = TimeFunction(name="u", grid=model.grid, time_order=time_order, space_order=space_order)
	pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
	stencil = Eq(u.forward, solve(pde, u.forward))
	src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m, offset=model.nbl)
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
	snaps = np.reshape(snaps, (number_of_snapshot, vnx, vnz))
	fobj.close()

	'''
	step 6: load computed snapshots from .npy files and plot wave propagation in each time step.
	'''

	plt.rcParams['figure.figsize'] = (20, 20)  # Increases figure size
	imcnt = 1 # Image counter for plotting
	number_of_save_snapshot = number_of_save_snapshot # Number of images to plot
	snap = '../snapshots/timestep_'
	for i in range(0, number_of_save_snapshot):
		imcnt = imcnt + 1
		ind = i * int(number_of_snapshot/number_of_save_snapshot)
		np.save(snap + str(imcnt), np.transpose(snaps[ind, :, :]))

	imcnt = 1 # Image counter for plotting
	for i in range(0, number_of_save_snapshot):
		imcnt = imcnt + 1
		ind = i * int(number_of_snapshot/number_of_save_snapshot)
		dummy = np.load(snap + str(imcnt) + '.npy')
		plt.imshow(dummy, cmap='gray')
		plt.title('Time Step: ' + str(imcnt))
		# save_file = ('timestep_' + str(imcnt))
		# plt.savefig('../snap_images/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
		plt.show()

def plot_processed_shot(data, perc, t0, tn):
	fig = plt.figure(figsize=(10, 10))
	data = scaling_velocity(data, -1, 1)
	max_num, min_num = clip(data, perc)
	extent = [0, data.shape[1]/100, 1e-3*tn, t0]
	plt.imshow(data, cmap='gray', vmin=min_num, vmax=max_num, extent=extent)
	plt.title('Recording Time 1500 ms')
	plt.xlabel('trace' + r'$\times 10^3$')
	plt.ylabel('time (s)')
	# plt.savefig('../image_out/' + 'r_1500' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def receiver_pipeline(space_order,
                      time_order,
                      image,
                      simulation_time_start,
                      simulation_time_stop,
					  source_frequency,
                      source_depth
                      ):

	'''
	step 1: define model size and geometry. For implicity, the model size is fixed by using nx x nx, 401 x 401. In other words, we use image having pixel x pixel, 401 x 401.
	'''

	# nx = 401; nz = 401; nb = 100
	nx = image.shape[1]; nz = image.shape[0]; nb = 100
	shape = (nx, nz)
	spacing = (6., 6.) # grid spacing
	origin = (0., 0.)
	model = Model(vp=np.transpose(image), origin=origin, shape=shape, spacing=spacing, space_order=space_order, nbl=400, bcs='damp')

	'''
	step 2: create layout for source and geophone acquisition
	'''

	dt = model.critical_dt  # Time step from model grid spacing
	time_range = TimeAxis(start=simulation_time_start, stop=simulation_time_stop, step=dt)
	nt = time_range.num  # number of time steps
	# NOTE source position
	src = RickerSource(name='src', grid=model.grid, f0=source_frequency, time_range=time_range)  
	# NOTE source layout
	src.coordinates.data[0, :] = np.array(model.domain_size) * .5
	src.coordinates.data[0, -1] = source_depth  # source depth
	# NOTE reciever positions
	rec = Receiver(name='rec', grid=model.grid, npoint=401, time_range=time_range)
	rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=401)
	rec.coordinates.data[:, 1] = source_depth  # receiver depth
	depth = rec.coordinates.data[:, 1]  
	geometry = AcquisitionGeometry(model, rec.coordinates.data, src.coordinates.data, simulation_time_start, simulation_time_stop, f0=source_frequency, src_type='Ricker')
	# NOTE plot ricker wavelet
	# geometry.src.show(); plt.show()
	# NOTE plot velocity model and acquisition layout
	# plot_velocity(model, source=src.coordinates.data, receiver=rec.coordinates.data[::4, :])
	# plt.show()

	# TODO solve forward modeling
	solver = AcousticWaveSolver(model, geometry, space_order=space_order)
	true_d , _, _ = solver.forward(vp=model.vp)
	plot_processed_shot(true_d.data, 99, simulation_time_start, simulation_time_stop)

def compute_kmeans(data, number_of_classes):
	vector_data = data.reshape(-1, 1) 
	random_centroid = 42 # interger number range 0-42
	kmeans = KMeans(n_clusters = number_of_classes, random_state = random_centroid).fit(vector_data)
	kmeans = kmeans.cluster_centers_[kmeans.labels_]
	kmeans = kmeans.reshape(data.shape)
	return kmeans 
