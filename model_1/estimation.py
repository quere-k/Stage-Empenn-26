import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize
import csv

"""
Function to evaluate the efficiency of the CAE and dictionary-based method
Estimation of the parameters from the selected subset using a least squares method
Comparison using statistics
"""

#collect the indices of the b-subset from a csv file
with open("./subset.csv", newline="") as mon_fichier:
    mon_fichier_reader = csv.reader(mon_fichier, delimiter=",")
    donnees = [[int(x) for x in row] for row in mon_fichier_reader]


nb_b=200 #number of b-values (cf N_input in CAE)
nb_D=80 #estimation on 80 D values
SNR=30 #signal noise ratio

#Generation of the b-subset
b_min, b_max = 10.0, 2000.0 # b-value (s/mm^2)
b = np.linspace(b_min,b_max,nb_b) #evenly spaced
subset=np.array(donnees)
subset = np.sort(np.unique(subset))
nb_subset=len(subset)
b_subset=b[subset]

#Generation of the diffusion coefficient
D_min, D_max = 0.1e-3, 3.0e-3 #min and max values of the diffusion coefficient
D = np.linspace(D_min,D_max,nb_D) #evenly spaced

#Signal generation using the subset
mat = np.zeros((nb_subset,nb_D))
for i in range(nb_subset):
    amplitude = 1/SNR
    noise = np.random.normal(0,amplitude)
    mat[i] = np.exp(-b_subset[i]*D)+noise

#Signal function
def monoExp(x, D):
    return np.exp(-D * x) 

#main function - estimation of parameters
D_est=np.array([])
for i in range(nb_D):
    p0 = (.001) #give a idea of the value
    params, cv = scipy.optimize.curve_fit(monoExp, b_subset, mat[:,i], p0)
    D_est=np.append(D_est,params)

error=np.abs(D_est-D)/D #absolute error
mean=np.mean(error) #mean of absolute error
std=np.std(error) #standard deviation of absolute error

#plot of the results
label=['D']
fig, ax = plt.subplots(figsize=(5, 6))
# Mean ± std
ax.bar(
    0, mean,
    yerr=std,
    capsize=6,
    width=0.5,
    color="lightsteelblue",
    edgecolor="black",
    alpha=0.8,
    label="Mean ± SD"
)
# Individual errors
x = np.random.normal(0, 0.04, len(error))  # horizontal jitter
ax.scatter(
    x,
    error,
    color="crimson",
    s=35,
    alpha=0.7,
    edgecolors="black",
    linewidth=0.3,
    label="Individual estimates"
)
ax.set_xticks([0])
ax.set_xticklabels(["D"])
ax.set_ylabel("Relative error")
ax.set_xlabel("Parameter")
ax.set_title("Relative error of the estimated diffusion coefficient\n(n = 80)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()


