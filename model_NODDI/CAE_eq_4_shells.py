import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from qspace.sampling.sphere import jones
import scipy.special
from scipy.special import lpmv
from math import pi, sqrt, tan, log
import numpy as np
import csv


class Erfi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = scipy.special.erfi(x.detach().cpu().numpy())
        return torch.as_tensor(y, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = (2.0 / sqrt(pi)) * torch.exp(x * x)
        return grad_output * grad

erfi = Erfi.apply

def check_nan(name, x):
    if not torch.isfinite(x).all():
        print(f"{name} has NaN/Inf")
        raise RuntimeError(name)

_GAMMA = 2.675987e8

class Signals(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self,idx):
        x=self.X[idx]
        y=self.Y[idx]
        return x,y

N_train=40
N_test=10
N_input=20
N_features=4
N_hidden_layer=2

SNR=30

nb_directions=30
directions=jones(nb_directions)
directions=torch.tensor(directions)

G_min, G_max= 30e-3, 65e-3  #T/m
G=torch.linspace(G_min, G_max, N_input)
G_dir=G*torch.ones((nb_directions,1)) #T/m

vol_iso = torch.rand(1, N_train).reshape(N_train, 1)
vol_ic = torch.rand(1, N_train).reshape(N_train, 1)
od_min, od_max=0.1, 0.9
od = od_min + (od_max-od_min)*torch.rand(1,N_train).reshape(N_train,1)
vol_iso = torch.squeeze(vol_iso)
vol_ic = torch.squeeze(vol_ic)
od = torch.squeeze(od)

# NODDI model - from AMICO
class NODDIIntraCellular: #Compute the signal in the intra-cellular compartment
    def __init__(self,grad_dirs, G, delta, smalldel):
        self.grad_dirs=grad_dirs
        self.G=G
        self.delta=delta
        self.smalldel=smalldel

    def get_signal(self, diff_par, od):
        kappa = 1/torch.tan((torch.pi*od)/2)
        check_nan("kappa", kappa)
        diff_par = diff_par * 1e-6
        return self._synth_meas_watson_SH_cyl_neuman_PGSE(
            torch.stack((diff_par, torch.tensor(0), kappa)),
            self.grad_dirs,
            torch.squeeze(self.G),
            torch.squeeze(self.delta),
            torch.squeeze(self.smalldel),
            torch.tensor([0, 0, 1]))

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
        ePerp = torch.exp(LePerp)
        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
        lgi = self._legendre_gaussian_integral(Lpmp, 6)
        # Compute the spherical harmonic coefficients of the Watson's distribution
        coeff = self._watson_SH_coeff(kappa)
        coeffMatrix = torch.tile(coeff, (l_q, 1))

        # Compute the dot product between the symmetry axis of the Watson's distribution and the gradient direction
        #
        # For numerical reasons, cosTheta might not always be between -1 and 1 due to round off errors, individual gradient vectors in grad_dirs and the fibredir are never exactly normal.  
        # When a gradient vector and fibredir are essentially parallel, their dot product can fall outside of -1 and 1.
        # BUT we need make sure it does, otherwise the legendre function call below will FAIL and abort the calculation!!!
        grad_dirs = grad_dirs.double()
        fibredir = fibredir.double()
        cosTheta = torch.matmul(grad_dirs,fibredir)
        badCosTheta = abs(cosTheta)>1
        cosTheta[badCosTheta] = cosTheta[badCosTheta]/abs(cosTheta[badCosTheta])

        # Compute the SH values at cosTheta
        sh = torch.zeros(coeff.shape)
        shMatrix = torch.tile(sh, (l_q, 1))
        for i in range(7):
            shMatrix[:,i] = sqrt((i+1 - .75)/pi)
            # legendre function returns coefficients of all m from 0 to l
            # we only need the coefficient corresponding to m = 0
            # WARNING: make sure to input ROW vector as variables!!!
            # cosTheta is expected to be a COLUMN vector.
            tmp = torch.zeros((l_q))
            for pol_i in range(l_q):
                tmp[pol_i] = lpmv(0, 2*i, cosTheta[pol_i]) #not on gpu
            shMatrix[:,i] = shMatrix[:,i]*tmp

        E = torch.sum(lgi*coeffMatrix*shMatrix, 1)
        # with the SH approximation, there will be no guarantee that E will be positive
        # but we need to make sure it does!!! replace the negative values with 10% of
        # the smallest positive values
        E[E<=0] = torch.min(E[E>0])*0.1
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
            LE = torch.zeros(G.shape) # np.size(R) = 1
            return LE
        else:
            ERROR( '"cyl_neuman_le_perp_PGSE" not yet validated for non-zero values' )

    def _legendre_gaussian_integral(self, Lpmp, n):
        if n > 6:
            ERROR( 'The maximum value for n is 6, which corresponds to the 12th order Legendre polynomial' )
        exact = Lpmp>0.05
        approx = Lpmp<=0.05

        mn = n + 1

        I = torch.zeros((len(Lpmp),mn))
        sqrtx = torch.sqrt(Lpmp[exact])
        I[exact,0] = sqrt(pi)*torch.erf(sqrtx)/sqrtx #erf support gradient
        dx = 1.0/Lpmp[exact]
        emx = -torch.exp(-Lpmp[exact])
        for i in range(1,mn):
            I[exact,i] = emx + (i-0.5)*I[exact,i-1]
            I[exact,i] = I[exact,i]*dx

        # Computing the legendre gaussian integrals for large enough Lpmp
        L = torch.zeros((len(Lpmp),n+1))
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
        x2=torch.float_power(Lpmp[approx],2)
        x3=x2*Lpmp[approx]
        x4=x3*Lpmp[approx]
        x5=x4*Lpmp[approx]
        x6=x5*Lpmp[approx]
        for i in range(n+1):
            if i == 0:
                L[approx,0] = (2 - 2*Lpmp[approx]/3 + x2/5 - x3/21 + x4/108).to(L.dtype)
            elif i == 1:
                L[approx,1] = (-4*Lpmp[approx]/15 + 4*x2/35 - 2*x3/63 + 2*x4/297).to(L.dtype)
            elif i == 2:
                L[approx,2] = (8*x2/315 - 8*x3/693 + 4*x4/1287).to(L.dtype)
            elif i == 3:
                L[approx,3] = (-16*x3/9009 + 16*x4/19305).to(L.dtype)
            elif i == 4:
                L[approx,4] = (32*x4/328185).to(L.dtype)
            elif i == 5:
                L[approx,5] = (-64*x5/14549535).to(L.dtype)
            elif i == 6:
                L[approx,6] = (128*x6/760543875).to(L.dtype)
        return L

    def _watson_SH_coeff(self, kappa):
        if isinstance(kappa,np.ndarray):
            ERROR( '"watson_SH_coeff()" not implemented for multiple kappa input yet' )

        # In the scope of AMICO only a single value is used for kappa
        n = 6

        C = torch.zeros((n+1))
        # 0th order is a constant
        C[0] = 2*sqrt(pi)

        # Precompute the special function values
        sk = torch.sqrt(kappa)
        sk2 = sk*kappa
        sk3 = sk2*kappa
        sk4 = sk3*kappa
        sk5 = sk4*kappa
        sk6 = sk5*kappa
        sk7 = sk6*kappa
        k2 = torch.float_power(kappa,2)
        k3 = k2*kappa
        k4 = k3*kappa
        k5 = k4*kappa
        k6 = k5*kappa
        k7 = k6*kappa

        erfik = erfi(sk)
        ierfik = 1/erfik
        ek = torch.exp(torch.clamp(kappa, max=50))
        dawsonk = 0.5*sqrt(pi)*erfik/ek

        if kappa > 0.1:

            # for large enough kappa
            C[1] = 3*sk - (3 + 2*kappa)*dawsonk
            C[1] = sqrt(5)*C[1]*ek
            C[1] = C[1]*ierfik/kappa

            C[2] = (105 + 60*kappa + 12*k2)*dawsonk
            C[2] = C[2] -105*sk + 10*sk2
            C[2] = .375*C[2]*ek/k2
            C[2] = C[2]*ierfik

            C[3] = -3465 - 1890*kappa - 420*k2 - 40*k3
            C[3] = C[3]*dawsonk
            C[3] = C[3] + 3465*sk - 420*sk2 + 84*sk3
            C[3] = C[3]*sqrt(13*pi)/64/k3
            C[3] = C[3]/dawsonk

            C[4] = 675675 + 360360*kappa + 83160*k2 + 10080*k3 + 560*k4
            C[4] = C[4]*dawsonk
            C[4] = C[4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
            C[4] = sqrt(17)*C[4]*ek
            C[4] = C[4]/512/k4
            C[4] = C[4]*ierfik

            C[5] = -43648605 - 22972950*kappa - 5405400*k2 - 720720*k3 - 55440*k4 - 2016*k5
            C[5] = C[5]*dawsonk
            C[5] = C[5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
            C[5] = sqrt(21*pi)*C[5]/4096/k5
            C[5] = C[5]/dawsonk

            C[6] = 7027425405 + 3666482820*kappa + 872972100*k2 + 122522400*k3  + 10810800*k4 + 576576*k5 + 14784*k6
            C[6] = C[6]*dawsonk
            C[6] = C[6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
            C[6] = 5*C[6]*ek
            C[6] = C[6]/16384/k6
            C[6] = C[6]*ierfik

        # for very large kappa
        if kappa>30:
            lnkd = log(kappa) - log(30)
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
            C[1] = C[1]*sqrt(pi/5)

            C[2] = 8/21*k2 + 32/693*k3
            C[2] = C[2]*(sqrt(pi)*0.2)

            C[3] = 16/693*k3 + 32/10395*k4
            C[3] = C[3]*sqrt(pi/13)

            C[4] = 32/19305*k4
            C[4] = C[4]*sqrt(pi/17)

            C[5] = 64*sqrt(pi/21)*k5/692835

            C[6] = 128*sqrt(pi)*k6/152108775
        return C

class NODDIExtraCellular:
    def __init__(self,grad_dirs, G, delta, smalldel):
        self.grad_dirs=grad_dirs
        self.G=G
        self.delta=delta
        self.smalldel=smalldel

    def get_signal(self, diff_par, od, vol_ic):
        diff_par = diff_par * 1e-6
        kappa = 1/torch.tan((torch.pi*od)/2)
        check_nan("kappa", kappa)
        diff_perp = diff_par * (1 - vol_ic)
        return self._synth_meas_watson_hindered_diffusion_PGSE(
            torch.stack((diff_par, diff_perp, kappa)),
            self.grad_dirs,
            torch.squeeze(self.G),
            torch.squeeze(self.delta),
            torch.squeeze(self.smalldel),
            torch.tensor([0, 0, 1]))

    # Extra-cellular signal
    def _synth_meas_watson_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]

        # get the equivalent diffusivities
        dw = self._watson_hindered_diffusion_coeff(dPar, dPerp, kappa)

        xh = torch.stack([dw[0], dw[1]])

        E = self._synth_meas_hindered_diffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)
        return E

    def _watson_hindered_diffusion_coeff(self, dPar, dPerp, kappa):
        dw = torch.zeros(2)
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2.*dPerp
            k2 = kappa*kappa
            dw[0] = dParP2dPerp/3.0 + 4.0*dParMdPerp*kappa/45.0 + 8.0*dParMdPerp*k2/945.0
            dw[1] = dParP2dPerp/3.0 - 2.0*dParMdPerp*kappa/45.0 - 4.0*dParMdPerp*k2/945.0
        else:
            eps = 1e-8
            kappa = kappa.clamp(max=100.0)
            sk = torch.sqrt(kappa)
            dawsonf = 0.5 * torch.exp(-kappa) * sqrt(pi) * erfi(sk)
            dawsonf = torch.clamp(dawsonf, min=eps)
            check_nan("dawsonf",dawsonf)
            factor = sk/dawsonf
            check_nan("factor", factor)
            dw[0] = (-dParMdPerp+2.0*dPerp*kappa+dParMdPerp*factor)/(2.0*kappa)
            dw[1] = (dParMdPerp+2.0*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4.0*kappa)
        return dw

    def _synth_meas_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar=x[0]
        dPerp=x[1]

        # Radial wavenumbers
        modQ = _GAMMA*smalldel*G
        modQ_Sq = torch.float_power(modQ,2.0)

        # Angles between gradient directions and fibre direction.
        grad_dirs = grad_dirs.double()
        fibredir = fibredir.double()
        cosTheta = torch.matmul(grad_dirs,fibredir)
        cosThetaSq = torch.float_power(cosTheta,2.0)
        sinThetaSq = 1.0-cosThetaSq

        # b-value
        bval = (delta-smalldel/3.0)*modQ_Sq

        # Find hindered signals
        E=torch.exp(-bval*((dPar - dPerp)*cosThetaSq + dPerp))
        return E

class NODDIIsotropic:
    def __init__(self,grad_dirs, G, delta, smalldel):
        self.grad_dirs=grad_dirs
        self.G=G
        self.delta=delta
        self.smalldel=smalldel

    def get_signal(self, diff_iso):
        diff_iso = diff_iso*1e-6
        return self._synth_meas_iso_GPD(diff_iso, self.smalldel, self.G, self.delta)

    # Isotropic signal
    def _synth_meas_iso_GPD(self, d, smalldel, gradient_strength, delta):
        modQ = _GAMMA*smalldel.transpose(0,1)*gradient_strength
        modQ_Sq = torch.float_power(modQ,2)
        difftime = delta.transpose(0,1)-smalldel.transpose(0,1)/3.0
        return torch.exp(-difftime*modQ_Sq*d)

def signal(G_dir, vol_iso, vol_ic, od, nb_directions):
    size=len(G_dir[0])
    signals=torch.zeros((size,nb_directions))
    delta=37.8e-3
    delta_dir=delta*torch.ones((nb_directions,1)) #s
    smalldel=17.5e-3
    smalldel_dir=smalldel*torch.ones((nb_directions,1)) #s
    d_par=torch.tensor(1.7e-3)
    d_iso=3.0e-3
    for i in range(size):
        G=G_dir[:,i]
        ic=NODDIIntraCellular(directions, G, delta_dir, smalldel_dir)
        signal_ic=ic.get_signal(d_par, od)
        check_nan("intra",signal_ic)
        ec=NODDIExtraCellular(directions, G, delta_dir, smalldel_dir)
        signal_ec=ec.get_signal(d_par, od, vol_ic)
        check_nan("extra",signal_ec)
        iso=NODDIIsotropic(directions, G, delta_dir, smalldel_dir)
        signal_iso=iso.get_signal(d_iso)
        check_nan("iso",signal_iso)
        signal=(1-vol_iso)*(vol_ic*signal_ic+(1-vol_ic)*signal_ec)+vol_iso*signal_iso
        signals[i]=signal_ec[:]
    return signals

A=torch.zeros((N_train, N_input, nb_directions))
for i in range(N_train):
    A[i]=signal(G_dir,vol_iso[i],vol_ic[i],od[i],nb_directions)

print("NaN:", torch.isnan(A).sum().item())
print("Inf:", torch.isinf(A).sum().item())
print("Finite:", torch.isfinite(A).all())

sigma=1/SNR
noise=torch.normal(0,sigma, size=(N_train,N_input, nb_directions))
X = A + noise

print("NaN:", torch.isnan(X).sum().item())
print("Inf:", torch.isinf(X).sum().item())
print("Finite:", torch.isfinite(X).all())
check_nan("train X",X)

training_data=Signals(X,X)

vol_iso = torch.rand(1, N_test).reshape(N_test, 1)
vol_ic = torch.rand(1, N_test).reshape(N_test, 1)
od_min, od_max=0.1, 0.9
od = od_min + (od_max-od_min)*torch.rand(1,N_test).reshape(N_test,1)
vol_iso = torch.squeeze(vol_iso)
vol_ic = torch.squeeze(vol_ic)
od = torch.squeeze(od)

A=torch.zeros((N_test, N_input, nb_directions))
for i in range(N_test):
    A[i]=signal(G_dir,vol_iso[i],vol_ic[i],od[i],nb_directions)

sigma=1/SNR
noise=torch.normal(0,sigma, size=(N_test,N_input, nb_directions))
Y = A + noise

check_nan("test X",Y)

testing_data=Signals(Y,Y)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

class ConcreteLayer(nn.Module):
    def __init__(self, num_inputs, num_features, pi_dropout=0.0):
        super().__init__()
        #Store parameters
        self.num_inputs=num_inputs
        self.num_features=num_features

        #Initialization
        logits_init=torch.rand(num_features,num_inputs)
        logits_init=logits_init/torch.sum(logits_init)
        self.logits=nn.Parameter(logits_init, requires_grad=True) #Learnable for the model
        self.pi_dropout=nn.Dropout(pi_dropout) #Desactivate inputs with pi=0.0, initialize Dropout layer
    
    def get_pi(self, ):

        logits=self.pi_dropout(self.logits)#Change logits values according to dropout
        pi=F.softmax(logits, dim=1)
        return pi, logits
    
    def sample_matrix(self, temperature, random, threshold, hard=False):
        pi, logits = self.get_pi()
        reg=self.regularization(logits, threshold) #calculate regularization function

        if not random:
            inds=torch.argmax(pi, dim=1) #Selection of the highest probability
            pi_deterministic=torch.zeros_like(pi) #Tensor with same dim as pi
            pi_deterministic[torch.arange(pi.shape[0]),inds]=1 #Put 1 at the right places
            selector_matrix=pi_deterministic #Matrix with 1 at selected channels
        else:
            selector_matrix=F.gumbel_softmax(logits, tau=temperature, hard=hard) #concrete distribution application
        return selector_matrix, reg
    
    def regularization(self, logits, threshold):
        num_inputs=self.num_inputs
        pi=F.softmax(logits, dim=1)
        L=torch.zeros(num_inputs)
        for i in range(num_inputs):
            L[i]=F.relu(torch.sum(pi[:,i]-threshold))
        reg=torch.sum(L)
        return reg

    def forward(self, x, random, temperature, threshold, hard=False):
        selector,reg=self.sample_matrix(temperature, random, threshold, hard)
        x = x.to(torch.float32)
        x = x.transpose(1, 2)
        x=F.linear(x,selector)
        outputs= {"latent": x, "reg": reg, "idx": torch.argmax(selector, dim=1)}
        return outputs

class CAE(nn.Module):
    def __init__(self, input_dim=N_input, features=N_features, n_hidden_layers=N_hidden_layer, dropout=0.0):
        super().__init__()
        indices2=np.arange(2+n_hidden_layers)
        data_indices2=np.array([indices2[0], indices2[-1]])
        data2=np.array([features,3])
        layer_sizes=np.interp(indices2, data_indices2, data2).astype(int)
        n_layers=len(layer_sizes)
        layers=[]
        for i in range(1, n_layers):
            if i==n_layers-1:
                layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
                layers.append(nn.LayerNorm(layer_sizes[i]))
                layers.append(nn.Softplus())
            else:
                layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
                layers.append(nn.LayerNorm(layer_sizes[i]))
                layers.append(nn.ReLU(True))
            
        print(layer_sizes,layers)
        self.encoder=ConcreteLayer(input_dim, features)
        self.decoder=nn.Sequential(*layers)

    def forward(self, x, random, temperature, threshold):
        eps = 1e-3
        check_nan("x",x)
        outputs=self.encoder(x, random, temperature, threshold)
        check_nan("latent", outputs["latent"])
        x = self.decoder(outputs["latent"])   # (..., 3)
        vf_iso = torch.sigmoid(x[:,:, 0]).clamp(eps, 1 - eps)
        vf_ic  = torch.sigmoid(x[:,:, 1]).clamp(eps, 1 - eps)
        od     = torch.sigmoid(x[:,:, 2]).clamp(eps, 1 - eps)
        check_nan("vf_iso", vf_iso)
        check_nan("vf_ic", vf_ic)
        check_nan("od", od)
        params = torch.stack((vf_iso, vf_ic, od), dim=-1)
        params=params.transpose(1,2)
        reg=outputs["reg"]
        returns = {'Parameters': params, 'REG': reg, 'Idx': outputs["idx"]}
        return returns

def temp_value(num_epochs, temp_base, temp_min, epoch):
    temp=temp_base*(temp_min/temp_base)**(epoch/num_epochs)
    return temp 
    
model=CAE()

learning_rate = 1e-3
batch_size = 64
epochs = 20

temp_base=10
temp_min=0.1
threshold=0.1
strength=0.1

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(epoch, model, train_loader, optimizer):
    print(f"train loop {epoch}")
    torch.autograd.set_detect_anomaly(True)
    model.train()
    sum_loss=0
    num_batches=len(train_loader)
    for batch, (X, y) in enumerate(train_loader):
        check_nan("X",X)
        temp=temp_value(epochs, temp_base, temp_min, epoch)
        optimizer.zero_grad()
        returns = model(X, True, temp, threshold)
        #reg=returns['REG']
        recon_batch=returns['Parameters']
        recon_batch=recon_batch.mean(axis=2)
        # print(recon_batch.requires_grad, recon_batch.grad_fn)
        # print(recon_batch.shape)
        size=len(recon_batch)
        recon_sign = []
        for i in range(size):
            vf_iso, vf_ic, od = recon_batch[i]
            recon_sign.append(signal(G_dir, vf_iso, vf_ic, od, nb_directions))
        recon_sign = torch.stack(recon_sign)
        # print(recon_sign.requires_grad)
        # print(recon_sign.grad_fn)
        loss = loss_function(recon_sign, y)
        sum_loss+=loss
        loss.backward()
        optimizer.step()
    sum_loss/=num_batches
    print(sum_loss)
    if epoch%5 ==0:
        print(f"Epoch {epoch}, Average Loss: {sum_loss:.6f}")

def test_loop(epoch, dataloader, model, loss_fn, loss_threshold,indices):
    print(f"test loop {epoch}")
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    absolute_errors = []
    with torch.no_grad():
        for X, y in dataloader:
            temp = temp_value(epochs, temp_base, temp_min, epoch)
            returns = model(X, False, temp, threshold)
            pred = returns['Parameters']
            idx = returns['Idx']
            recon_batch=pred.mean(axis=2)
            size=len(recon_batch)
            recon_sign = []
            for i in range(size):
                vf_iso, vf_ic, od = recon_batch[i]
                recon_sign.append(signal(G_dir, vf_iso, vf_ic, od, nb_directions))
            recon_sign = torch.stack(recon_sign)
            test_loss += loss_fn(recon_sign, y).item()
            for i in range(len(recon_sign)):
                for j in range(N_input):
                    error = abs(recon_sign[i, j] - y[i, j])
                    absolute_errors.append(error)
    test_loss/=num_batches
    mean_ae = sum(absolute_errors) / len(absolute_errors)
    mean_ae = torch.mean(mean_ae)
    max_ae = torch.cat(absolute_errors).max()
    min_ae = torch.cat(absolute_errors).min()
    print(test_loss)
    if test_loss<loss_threshold:
        indices=idx.tolist().copy()
        loss_threshold=test_loss
    if epoch%5==0:
        print(
                f"Test Error: \n"
                f"Avg loss: {test_loss:>8f} \n"
                f"Absolute Error - Mean: {mean_ae} - Min: {min_ae} - Max: {max_ae}"
            )
    return indices, loss_threshold

indices = None
loss_threshold = float('inf')  # Initialisation

for epoch in range(1, epochs + 1):
    train_loop(epoch, model, train_dataloader, optimizer)
    indices, loss_threshold = test_loop(epoch, test_dataloader, model, loss_function,loss_threshold,indices)

indices=np.array(indices)

best_G=G[indices]
best_G=np.array(best_G)

# def b_value(G):
#     delta=37.8e-3
#     smalldel=17.5e-3
#     modQ = _GAMMA*smalldel*G
#     modQ_Sq = np.power(modQ,2)
#     difftime = delta-smalldel/3.0
#     return difftime*modQ_Sq/np.power(10,6)

# b_val=b_value(best_G)

chemin = "./subset_eq_4_shells.csv"

with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

    mon_fichier_ecrire.writerow(best_G)