import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from scipy import ndimage

def plot_velocity(model, title):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
	# NOTE convert model size to km by dividing 100
	extent=[0, model.shape[1]/100, model.shape[0]/100, 0] 
	rotated_img = ndimage.rotate(model, 90)
	cax = ax.imshow(rotated_img, cmap='jet', extent=extent)
	ax.xaxis.set_label_position('bottom') 
	ax.xaxis.tick_bottom()
	ax.set_xlabel('distance (km)')
	ax.set_ylabel('depth (km)')
	plt.title(title)
	cbar = fig.colorbar(cax, orientation='vertical', fraction=0.047, pad=0.01, shrink=1.0)
	cbar.set_label('velocity (km/s)', labelpad=18, rotation=270)
	# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()