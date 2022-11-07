import numpy as np
import matplotlib.pyplot as plt

'''
forward modeling of a buried sphere (single observation)
'''

G = pow(6.67384, -11)
rho_host = 2100 # shale (kg/m3)
rho_target = 2700 # granite (kg/m3)
R = 100 # meters 
z = 200 # meters
x = 500 # meters

def gravity_sphere(G, rho_host, rho_target, R, z, x):
	x1 = ( (rho_target - rho_host) * ((4/3) * np.pi * pow(R, 3)) )
	x2 = G*z*x1
	x3 = pow(pow(x, 2) + pow(z, 2), 1.5)
	gz = x2/x3
	gal = gz*0.01 # unit gal
	mgal = gal*pow(10, 6) # unit gal
	return mgal

mgal_gz = gravity_sphere(G, rho_host, rho_target, R, z, x)

print(mgal_gz)

'''
forward modeling of a buried sphere (multiple observatal positions)
'''

number_of_stations = 50
x = np.linspace(-500, 500, number_of_stations) # meters
fig = plt.figure(figsize=(10, 10))
for index, i in enumerate(x):
	mgal_gz = gravity_sphere(G, rho_host, rho_target, R, z, i)
	print(mgal_gz)
	plt.scatter(i, mgal_gz, s=100, color='#CB2B0B', edgecolors='black', alpha=1.0, marker='o')
	plt.xlim(-510, 510)
	plt.ylim(0.024, 0.55)
	# dummy.append(mgal_gz)
	# plt.show()