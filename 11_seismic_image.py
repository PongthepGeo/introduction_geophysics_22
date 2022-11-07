import sys
sys.path.append('./Libs') 
import function as f
#-----------------------------------------------------------------------------------------#
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
#-----------------------------------------------------------------------------------------#
# pip install scikit-image
# pip install scikit-learn
# pip install opencv-python
#-----------------------------------------------------------------------------------------#

'''
step 1: image conversion
'''

# img = imageio.imread('datasets/graben.png') / 255
# # plt.imshow(img)
# # plt.show()
# _, smooth = f.convert2velocity(img, img.shape[0], img.shape[1], 5)
# seismic = f.reflectivity(smooth, 15)
# max_num, min_num = f.clip(seismic, 99.8)
# # plt.imshow(seismic, vmin=min_num, vmax=max_num, cmap='gray')
# # plt.show()
# plt.imsave(fname='image_out/seismic.png', arr=seismic, cmap='gray', format='png', vmin=min_num, vmax=max_num)

'''
step 2: 
'''

img = imageio.imread('datasets/graben.png') / 255
img = img[:, :, 0]
ROW, COL = img.shape

fig = plt.figure(figsize=(25, 15))

ax1 = fig.add_subplot(2, 2, 1)
# to draw a line from (205, 0) to (205, 720 )
x = [205, 205]
y = [0, ROW]
ax1.plot(x, y, color='red', linewidth=3, linestyle='--')
ax1.imshow(img, aspect='auto')
ax1.title.set_text('Original image')
ax1.set_xlabel('pixel-x')
ax1.set_ylabel('pixel-y')

ax2 = fig.add_subplot(2, 2, 2)
axis_y = np.linspace(0, ROW, ROW)
ax2.plot(img[:, 205], axis_y, color='red', linewidth=1, linestyle='-')
ax2.title.set_text('Signal at column 205')
ax2.invert_yaxis()
ax2.set_xlabel('amplitude')
ax2.set_ylabel('pixel-y')
ax2.set_ylim(ROW, 0)

# NOTE load reflectivity
ref = imageio.imread('image_out/seismic.png') / 255

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, y, color='green', linewidth=3, linestyle='--')
ax3.imshow(ref, aspect='auto')
ax3.title.set_text('Blur image')
ax3.set_xlabel('pixel-x')
ax3.set_ylabel('pixel-y')

ax4 = fig.add_subplot(2, 2, 4)
axis_y = np.linspace(0, ROW, ROW)
ax4.plot(ref[:, 205], axis_y, color='green', linewidth=1, linestyle='-')
ax4.title.set_text('Signal at column 205')
ax4.invert_yaxis()
ax4.set_xlabel('amplitude')
ax4.set_ylabel('pixel-y')
ax4.set_ylim(ROW, 0)

plt.savefig('image_out/seismic_img' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()