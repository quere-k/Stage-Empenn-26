import numpy as np 
from qspace.sampling.sphere import jones
from scipy.special import erf, erfi, lpmv, dawsn
import csv
import ast

""" 
Function to obtain an optimized subset of acquisitions using permutation
Signals are computed from G values with the same number of directions
4 values are selected
Based on the NODDI model
"""

#retrieve parameter triplets
with open("./param_2.csv", newline='') as f:
    reader = csv.reader(f)

    vf_iso = [ast.literal_eval(x) for x in next(reader)]
    vf_ic  = [ast.literal_eval(x) for x in next(reader)]
    od  = [ast.literal_eval(x) for x in next(reader)]

_GAMMA = 2.675987e8

#shell directions
nb_directions=30
directions=jones(nb_directions)

vf_iso=np.array(vf_iso)
vf_ic=np.array(vf_ic)
od=np.array(od)

#G values generation
G_min, G_max= 30e-3, 65e-3  #T/m
G=np.linspace(G_min, G_max, 20) #evenly spaced
G_dir=G*np.ones((nb_directions,1)) #T/m
kappa=1/np.tan((np.pi*od)/2)

#AMICO classes - NODDI signals
class NODDIIntraCellular: #Compute the signal in the intra-cellular compartment
    def __init__(self,grad_dirs, G, delta, smalldel):
        self.grad_dirs=grad_dirs
        self.G=G
        self.delta=delta
        self.smalldel=smalldel

    def get_signal(self, diff_par, kappa):
        diff_par *= 1e-6
        return self._synth_meas_watson_SH_cyl_neuman_PGSE(
            np.array([diff_par, 0, kappa]),
            self.grad_dirs,
            np.squeeze(self.G),
            np.squeeze(self.delta),
            np.squeeze(self.smalldel),
            np.array([0, 0, 1]))

    # Intra-cellular signal
    def _synth_meas_watson_SH_cyl_neuman_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        d=x[0]
        R=x[1]
        kappa=x[2]

        l_q = grad_dirs.shape[0]

        # Parallel component
        LePar = self._cyl_neuman_le_par_PGSE(d, G, delta, smalldel)
        # Perpendicular component
        LePerp = self._cyl_neuman_le_perp_PGSE(d, R, G)
        ePerp = np.exp(LePerp)
        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
        lgi = self._legendre_gaussian_integral(Lpmp, 6)
        # Compute the spherical harmonic coefficients of the Watson's distribution
        coeff = self._watson_SH_coeff(kappa)
        coeffMatrix = np.tile(coeff, (l_q, 1))

        # Compute the dot product between the symmetry axis of the Watson's distribution and the gradient direction
        #
        # For numerical reasons, cosTheta might not always be between -1 and 1 due to round off errors, individual gradient vectors in grad_dirs and the fibredir are never exactly normal.  
        # When a gradient vector and fibredir are essentially parallel, their dot product can fall outside of -1 and 1.
        # BUT we need make sure it does, otherwise the legendre function call below will FAIL and abort the calculation!!!
        cosTheta = np.dot(grad_dirs,fibredir)
        badCosTheta = abs(cosTheta)>1
        cosTheta[badCosTheta] = cosTheta[badCosTheta]/abs(cosTheta[badCosTheta])

        # Compute the SH values at cosTheta
        sh = np.zeros(coeff.shape)
        shMatrix = np.tile(sh, (l_q, 1))
        for i in range(7):
            shMatrix[:,i] = np.sqrt((i+1 - .75)/np.pi)
            # legendre function returns coefficients of all m from 0 to l
            # we only need the coefficient corresponding to m = 0
            # WARNING: make sure to input ROW vector as variables!!!
            # cosTheta is expected to be a COLUMN vector.
            tmp = np.zeros((l_q))
            for pol_i in range(l_q):
                tmp[pol_i] = lpmv(0, 2*i, cosTheta[pol_i])
            shMatrix[:,i] = shMatrix[:,i]*tmp

        E = np.sum(lgi*coeffMatrix*shMatrix, 1)
        # with the SH approximation, there will be no guarantee that E will be positive
        # but we need to make sure it does!!! replace the negative values with 10% of
        # the smallest positive values
        E[E<=0] = np.min(E[E>0])*0.1
        E = 0.5*E*ePerp
        return E

    def _cyl_neuman_le_par_PGSE(self, d, G, delta, smalldel):
        # Line bellow used in matlab version removed as cyl_neuman_le_par_PGSE is called from synth_meas_watson_SH_cyl_neuman_PGSE which already casts x to d, R and kappa -> x replaced by d in arguments
        #d=x[0]

        # Radial wavenumbers
        modQ = _GAMMA*smalldel*G
        modQ_Sq = modQ*modQ

        # diffusion time for PGSE, in a matrix for the computation below.
        difftime = (delta-smalldel/3)

        # Parallel component
        LE =-modQ_Sq*difftime*d

        # Compute the Jacobian matrix
        #if(nargout>1)
        #    % dLE/d
        #    J = -modQ_Sq*difftime
        #end
        return LE

    def _cyl_neuman_le_perp_PGSE(self, d, R, G):
        # When R=0, no need to do any calculation
        if (R == 0.00):
            LE = np.zeros(G.shape) # np.size(R) = 1
            return LE
        else:
            ERROR( '"cyl_neuman_le_perp_PGSE" not yet validated for non-zero values' )

    def _legendre_gaussian_integral(self, Lpmp, n):
        if n > 6:
            ERROR( 'The maximum value for n is 6, which corresponds to the 12th order Legendre polynomial' )
        exact = Lpmp>0.05
        approx = Lpmp<=0.05

        mn = n + 1

        I = np.zeros((len(Lpmp),mn))
        sqrtx = np.sqrt(Lpmp[exact])
        I[exact,0] = np.sqrt(np.pi)*erf(sqrtx)/sqrtx
        dx = 1.0/Lpmp[exact]
        emx = -np.exp(-Lpmp[exact])
        for i in range(1,mn):
            I[exact,i] = emx + (i-0.5)*I[exact,i-1]
            I[exact,i] = I[exact,i]*dx

        # Computing the legendre gaussian integrals for large enough Lpmp
        L = np.zeros((len(Lpmp),n+1))
        for i in range(n+1):
            if i == 0:
                L[exact,0] = I[exact,0]
            elif i == 1:
                L[exact,1] = -0.5*I[exact,0] + 1.5*I[exact,1]
            elif i == 2:
                L[exact,2] = 0.375*I[exact,0] - 3.75*I[exact,1] + 4.375*I[exact,2]
            elif i == 3:
                L[exact,3] = -0.3125*I[exact,0] + 6.5625*I[exact,1] - 19.6875*I[exact,2] + 14.4375*I[exact,3]
            elif i == 4:
                L[exact,4] = 0.2734375*I[exact,0] - 9.84375*I[exact,1] + 54.140625*I[exact,2] - 93.84375*I[exact,3] + 50.2734375*I[exact,4]
            elif i == 5:
                L[exact,5] = -(63./256.)*I[exact,0] + (3465./256.)*I[exact,1] - (30030./256.)*I[exact,2] + (90090./256.)*I[exact,3] - (109395./256.)*I[exact,4] + (46189./256.)*I[exact,5]
            elif i == 6:
                L[exact,6] = (231./1024.)*I[exact,0] - (18018./1024.)*I[exact,1] + (225225./1024.)*I[exact,2] - (1021020./1024.)*I[exact,3] + (2078505./1024.)*I[exact,4] - (1939938./1024.)*I[exact,5] + (676039./1024.)*I[exact,6]

        # Computing the legendre gaussian integrals for small Lpmp
        x2=np.power(Lpmp[approx],2)
        x3=x2*Lpmp[approx]
        x4=x3*Lpmp[approx]
        x5=x4*Lpmp[approx]
        x6=x5*Lpmp[approx]
        for i in range(n+1):
            if i == 0:
                L[approx,0] = 2 - 2*Lpmp[approx]/3 + x2/5 - x3/21 + x4/108
            elif i == 1:
                L[approx,1] = -4*Lpmp[approx]/15 + 4*x2/35 - 2*x3/63 + 2*x4/297
            elif i == 2:
                L[approx,2] = 8*x2/315 - 8*x3/693 + 4*x4/1287
            elif i == 3:
                L[approx,3] = -16*x3/9009 + 16*x4/19305
            elif i == 4:
                L[approx,4] = 32*x4/328185
            elif i == 5:
                L[approx,5] = -64*x5/14549535
            elif i == 6:
                L[approx,6] = 128*x6/760543875
        return L

    def _watson_SH_coeff(self, kappa):
        if isinstance(kappa,np.ndarray):
            ERROR( '"watson_SH_coeff()" not implemented for multiple kappa input yet' )

        # In the scope of AMICO only a single value is used for kappa
        n = 6

        C = np.zeros((n+1))
        # 0th order is a constant
        C[0] = 2*np.sqrt(np.pi)

        # Precompute the special function values
        sk = np.sqrt(kappa)
        sk2 = sk*kappa
        sk3 = sk2*kappa
        sk4 = sk3*kappa
        sk5 = sk4*kappa
        sk6 = sk5*kappa
        sk7 = sk6*kappa
        k2 = np.power(kappa,2)
        k3 = k2*kappa
        k4 = k3*kappa
        k5 = k4*kappa
        k6 = k5*kappa
        k7 = k6*kappa

        erfik = erfi(sk)
        ierfik = 1/erfik
        ek = np.exp(kappa)
        dawsonk = dawsn(sk)

        if kappa > 0.1:

            # for large enough kappa
            C[1] = 3*sk - (3 + 2*kappa)*dawsonk
            C[1] = np.sqrt(5)*C[1]*ek
            C[1] = C[1]*ierfik/kappa

            C[2] = (105 + 60*kappa + 12*k2)*dawsonk
            C[2] = C[2] -105*sk + 10*sk2
            C[2] = .375*C[2]*ek/k2
            C[2] = C[2]*ierfik

            C[3] = -3465 - 1890*kappa - 420*k2 - 40*k3
            C[3] = C[3]*dawsonk
            C[3] = C[3] + 3465*sk - 420*sk2 + 84*sk3
            C[3] = C[3]*np.sqrt(13*np.pi)/64/k3
            C[3] = C[3]/dawsonk

            C[4] = 675675 + 360360*kappa + 83160*k2 + 10080*k3 + 560*k4
            C[4] = C[4]*dawsonk
            C[4] = C[4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
            C[4] = np.sqrt(17)*C[4]*ek
            C[4] = C[4]/512/k4
            C[4] = C[4]*ierfik

            C[5] = -43648605 - 22972950*kappa - 5405400*k2 - 720720*k3 - 55440*k4 - 2016*k5
            C[5] = C[5]*dawsonk
            C[5] = C[5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
            C[5] = np.sqrt(21*np.pi)*C[5]/4096/k5
            C[5] = C[5]/dawsonk

            C[6] = 7027425405 + 3666482820*kappa + 872972100*k2 + 122522400*k3  + 10810800*k4 + 576576*k5 + 14784*k6
            C[6] = C[6]*dawsonk
            C[6] = C[6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
            C[6] = 5*C[6]*ek
            C[6] = C[6]/16384/k6
            C[6] = C[6]*ierfik

        # for very large kappa
        if kappa>30:
            lnkd = np.log(kappa) - np.log(30)
            lnkd2 = lnkd*lnkd
            lnkd3 = lnkd2*lnkd
            lnkd4 = lnkd3*lnkd
            lnkd5 = lnkd4*lnkd
            lnkd6 = lnkd5*lnkd
            C[1] = 7.52308 + 0.411538*lnkd - 0.214588*lnkd2 + 0.0784091*lnkd3 - 0.023981*lnkd4 + 0.00731537*lnkd5 - 0.0026467*lnkd6
            C[2] = 8.93718 + 1.62147*lnkd - 0.733421*lnkd2 + 0.191568*lnkd3 - 0.0202906*lnkd4 - 0.00779095*lnkd5 + 0.00574847*lnkd6
            C[3] = 8.87905 + 3.35689*lnkd - 1.15935*lnkd2 + 0.0673053*lnkd3 + 0.121857*lnkd4 - 0.066642*lnkd5 + 0.0180215*lnkd6
            C[4] = 7.84352 + 5.03178*lnkd - 1.0193*lnkd2 - 0.426362*lnkd3 + 0.328816*lnkd4 - 0.0688176*lnkd5 - 0.0229398*lnkd6
            C[5] = 6.30113 + 6.09914*lnkd - 0.16088*lnkd2 - 1.05578*lnkd3 + 0.338069*lnkd4 + 0.0937157*lnkd5 - 0.106935*lnkd6
            C[6] = 4.65678 + 6.30069*lnkd + 1.13754*lnkd2 - 1.38393*lnkd3 - 0.0134758*lnkd4 + 0.331686*lnkd5 - 0.105954*lnkd6

        if kappa <= 0.1:
            # for small kappa
            C[1] = 4/3*kappa + 8/63*k2
            C[1] = C[1]*np.sqrt(np.pi/5)

            C[2] = 8/21*k2 + 32/693*k3
            C[2] = C[2]*(np.sqrt(np.pi)*0.2)

            C[3] = 16/693*k3 + 32/10395*k4
            C[3] = C[3]*np.sqrt(np.pi/13)

            C[4] = 32/19305*k4
            C[4] = C[4]*np.sqrt(np.pi/17)

            C[5] = 64*np.sqrt(np.pi/21)*k5/692835

            C[6] = 128*np.sqrt(np.pi)*k6/152108775
        return C

class NODDIExtraCellular: #Compute the signal in the extra-cellular compartment
    def __init__(self,grad_dirs, G, delta, smalldel):
        self.grad_dirs=grad_dirs
        self.G=G
        self.delta=delta
        self.smalldel=smalldel

    def get_signal(self, diff_par, kappa, vol_ic):
        diff_par *= 1e-6
        diff_perp = diff_par * (1 - vol_ic)
        return self._synth_meas_watson_hindered_diffusion_PGSE(
            np.array([diff_par, diff_perp, kappa]),
            self.grad_dirs,
            np.squeeze(self.G),
            np.squeeze(self.delta),
            np.squeeze(self.smalldel),
            np.array([0, 0, 1]))

    # Extra-cellular signal
    def _synth_meas_watson_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]

        # get the equivalent diffusivities
        dw = self._watson_hindered_diffusion_coeff(dPar, dPerp, kappa)

        xh = np.array([dw[0], dw[1]])

        E = self._synth_meas_hindered_diffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)
        return E

    def _watson_hindered_diffusion_coeff(self, dPar, dPerp, kappa):
        dw = np.zeros(2)
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2.*dPerp
            k2 = kappa*kappa
            dw[0] = dParP2dPerp/3.0 + 4.0*dParMdPerp*kappa/45.0 + 8.0*dParMdPerp*k2/945.0
            dw[1] = dParP2dPerp/3.0 - 2.0*dParMdPerp*kappa/45.0 - 4.0*dParMdPerp*k2/945.0
        else:
            sk = np.sqrt(kappa)
            dawsonf = dawsn(sk)
            factor = sk/dawsonf
            dw[0] = (-dParMdPerp+2.0*dPerp*kappa+dParMdPerp*factor)/(2.0*kappa)
            dw[1] = (dParMdPerp+2.0*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4.0*kappa)
        return dw

    def _synth_meas_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar=x[0]
        dPerp=x[1]

        # Radial wavenumbers
        modQ = _GAMMA*smalldel*G
        modQ_Sq = np.power(modQ,2.0)

        # Angles between gradient directions and fibre direction.
        cosTheta = np.dot(grad_dirs,fibredir)
        cosThetaSq = np.power(cosTheta,2.0)
        sinThetaSq = 1.0-cosThetaSq

        # b-value
        bval = (delta-smalldel/3.0)*modQ_Sq

        # Find hindered signals
        E=np.exp(-bval*((dPar - dPerp)*cosThetaSq + dPerp))
        return E

class NODDIIsotropic: #Compute the signal in the CSF
    def __init__(self,grad_dirs, G, delta, smalldel):
        self.grad_dirs=grad_dirs
        self.G=G
        self.delta=delta
        self.smalldel=smalldel

    def get_signal(self, diff_iso):
        diff_iso *= 1e-6
        return self._synth_meas_iso_GPD(diff_iso, self.smalldel, self.G, self.delta)

    # Isotropic signal
    def _synth_meas_iso_GPD(self, d, smalldel, gradient_strength, delta):
        modQ = _GAMMA*smalldel.transpose()*gradient_strength.transpose()
        modQ_Sq = np.power(modQ,2)
        difftime = delta.transpose()-smalldel.transpose()/3.0
        return np.exp(-difftime*modQ_Sq*d)

#Acquisition parameters
d_par=1.7e-3
d_iso=3.0e-3
delta=37.8e-3
delta_dir=delta*np.ones((nb_directions,1)) #s
smalldel=17.5e-3
smalldel_dir=smalldel*np.ones((nb_directions,1)) #s
ic_model=[]
ec_model=[]
iso_model=[]
for j in range(20):
    Gj=G_dir[:,j]
    ic_model.append(NODDIIntraCellular(directions, Gj, delta_dir, smalldel_dir))
    ec_model.append(NODDIExtraCellular(directions, Gj, delta_dir, smalldel_dir))
    iso_model.append(NODDIIsotropic(directions, Gj, delta_dir, smalldel_dir))

def signal(params, ic_model, ec_model, iso_model): #compute the total signal
    """
    params:
        params[0] = vol_iso
        params[1] = vol_ic
        params[2] = od
    """
    vol_iso, vol_ic, od = params
    signal_ic = ic_model.get_signal(d_par, od)
    signal_ec = ec_model.get_signal(d_par, od, vol_ic)
    signal_iso = iso_model.get_signal(d_iso)
    S = (
        (1 - vol_iso)
        * (vol_ic * signal_ic + (1 - vol_ic) * signal_ec)
        + vol_iso * signal_iso
    )
    return np.asarray(S).transpose(0,1)

#signal matrix (20,345,30)
N=len(od)
mat = np.zeros((20,N,nb_directions))
for i in range(N):
    for j in range(20):
        mat[j][i] = signal((vf_iso[i],vf_ic[i],kappa[i]), ic_model[j], ec_model[j], iso_model[j])

def distance(mat): #function to compute the distance matrix between all 345 triplets for each value of G
    size, N, nb_dir = len(mat), len(mat[0]), len(mat[0][0])
    distance_matrices = []
    for k in range(size):
        distance = np.zeros((N, N, nb_dir))
        for i in range(N):
            for j in range(N):
                if j > i:
                    distance[i][j] = np.abs(mat[k][i] - mat[k][j])
        distance_matrices.append(distance)
    distance_matrix = np.stack(distance_matrices, axis=0)
    return distance_matrix

def score(distance_matrix, SNR): #function to compute the score of each value of G ie the number of signal pairs it can discriminate
    size, N, nb_dir= len(distance_matrix), len(distance_matrix[0]), len(distance_matrix[0][0][0])
    score_mat = np.array([])
    for k in range(size):
        score = 0
        for i in range(N):
            for j in range(N):
                if j>i:
                    for l in range(nb_dir):
                        if distance_matrix[k][i][j][l]>=(1/SNR):
                            score+=1
        score_mat=np.append(score_mat, score)
    return score_mat

distance_matrix=distance(mat)
print(f"We compute the distance matrix.")

def permutation(distance_matrix, nb_b, nb_rep): #function to obtain an optimal subset of gradient strength by permutation
    best_subset_seen = np.array([])
    best_nb_d = 0
    SNR = 25
    for i in range(nb_rep):
        print(f"\trepetition number {i+1}")
        subset = np.random.choice(20, nb_b, replace=False) #random choice of subset
        selection = distance_matrix[subset]
        score_mat = np.array([])
        changed = True
        while changed:
            score_mat = score(selection,SNR)
            weakest_b = subset[np.argmin(score_mat)] #G value with the lowest score 
            nb_weakest = np.sum(score_mat)
            next_best_b = weakest_b
            nb_next_best = 0
            while (nb_next_best < nb_weakest and next_best_b < len(distance_matrix) - 1): #find a better G value (highest score) and insert it in the subset
                next_best_b =  next_best_b + 1
                new_selection = selection.copy()
                new_selection[np.argmin(score_mat)] = distance_matrix[next_best_b]
                new_score = score(new_selection,SNR)
                nb_next_best = np.sum(new_score)
                if nb_weakest<nb_next_best and next_best_b not in subset:
                    selection = new_selection
                    subset[np.argmin(score_mat)]=next_best_b
                else:
                    changed = False
        nb_d_subset = nb_next_best
        if nb_d_subset>best_nb_d:
            best_nb_d = nb_d_subset
            best_subset_seen = subset.copy()
    return best_subset_seen,best_nb_d

nb_G=4
best_subset,best_nb=permutation(distance_matrix, nb_G, 6) #6 repetitions 
#retrieve the subset indice and the G values associated
best_G=G[best_subset]

#best subset in a csv file
chemin = f"./subset_permut_4_shells.csv"

with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

    mon_fichier_ecrire.writerow(best_G)
