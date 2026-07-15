import numpy as np 
import csv

b_min, b_max = 10.0, 2000.0 #s/mm^2
b = np.linspace(b_min,b_max,200)

D_min, D_max = 0.1e-3, 3.0e-3
D=np.array([D_min, D_max])

def division(D_min, D_max, D, b):
    SNR = 30
    S_min = np.exp(-b*D_min)
    S_max = np.exp(-b*D_max)
    distance = np.abs(S_min-S_max)
    if np.max(distance)>=(1/SNR):
        D_mean = (D_min+D_max)/2
        D=np.append(D,D_mean)
        D=division(D_min, D_mean, D, b)
        D=division(D_mean, D_max, D, b)
    return D

D=division(D_min, D_max, D, b)
D = np.sort(np.unique(D))

N = len(D)
print(N)
mat = np.zeros((200,N))
for i in range(len(b)):
    mat[i] = np.exp(-b[i]*D)

def distance(mat):
    size, N = len(mat), len(mat[0])
    distance_matrices = []
    for k in range(size):
        distance = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if j > i:
                    distance[i][j] = np.abs(mat[k][i] - mat[k][j])
        distance_matrices.append(distance)
    distance_matrix = np.stack(distance_matrices, axis=0)
    return distance_matrix

def score(distance_matrix, SNR):
    size, N = len(distance_matrix), len(distance_matrix[0])
    score_mat = np.array([])
    for k in range(size):
        score = 0
        for i in range(N):
            for j in range(N):
                if j>i:
                    if distance_matrix[k][i][j]>=(1/SNR):
                        score+=1
        score_mat=np.append(score_mat, score)
    return score_mat

distance_matrix=distance(mat)

def permutation(distance_matrix, nb_b, nb_rep):
    best_subset_seen = np.array([])
    best_nb_d = 0
    SNR = 25
    for i in range(nb_rep):
        subset =  np.random.randint(0,200,nb_b)
        selection = distance_matrix[subset]
        score_mat = np.array([])
        changed = True
        while changed:
            score_mat = score(selection,SNR)
            weakest_b = subset[np.argmin(score_mat)]
            nb_weakest = np.sum(score_mat)
            next_best_b = weakest_b
            nb_next_best = 0
            while (nb_next_best < nb_weakest and next_best_b < len(distance_matrix) - 1):
                next_best_b =  next_best_b + 1
                new_selection = selection.copy()
                new_selection[np.argmin(score_mat)] = distance_matrix[next_best_b]
                new_score = score(new_selection,SNR)
                nb_next_best = np.sum(new_score)
                if nb_weakest<nb_next_best:
                    selection = new_selection
                    subset[np.argmin(score_mat)]=next_best_b
                else:
                    changed = False
        nb_d_subset = nb_next_best
        if nb_d_subset>best_nb_d:
            best_nb_d = nb_d_subset
            best_subset_seen = subset.copy()
    return best_subset_seen

best_subset=permutation(distance_matrix, 20, 6)

chemin = "./subset.csv"

with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

    mon_fichier_ecrire.writerow(best_subset)

# print(best_subset, len(best_subset))
