import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize
import csv

with open("./subset_eq.csv", newline="") as mon_fichier:
    mon_fichier_reader = csv.reader(mon_fichier, delimiter=",")
    donnees = [[int(x) for x in row] for row in mon_fichier_reader]

nb_b=200
nb_D=11
SNR=30

b_min, b_max = 10.0, 2000.0 #s/mm^2
b = np.linspace(b_min,b_max,nb_b)

subset=np.array(donnees)
subset = np.sort(np.unique(subset))

nb_subset=len(subset)

b_subset=b[subset]

D_min, D_max = 0.1e-3, 3.0e-3
D = np.linspace(D_min,D_max,nb_D)

mat = np.zeros((nb_subset,nb_D))
for i in range(nb_subset):
    amplitude = 1/SNR
    noise = np.random.normal(0,amplitude)
    mat[i] = np.exp(-b_subset[i]*D)+noise

def monoExp(x, D):
    return np.exp(-D * x) 

D_est=np.array([])
for i in range(nb_D):
    p0 = (.001)
    params, cv = scipy.optimize.curve_fit(monoExp, b_subset, mat[:,i], p0)
    D_est=np.append(D_est,params)

print(D_est,D)

error=np.abs(D_est-D)/D
print(100*error)

plt.plot(b_subset, mat[:,0], '.', label="data")
plt.plot(b_subset, monoExp(b_subset,D_est[0]), '--', label="fitted")
plt.title("Fitted dMRI Signal")
plt.xlabel("b-values")
plt.xlim(0,2000)
plt.ylabel("MRI signal")
plt.grid()
plt.show()


