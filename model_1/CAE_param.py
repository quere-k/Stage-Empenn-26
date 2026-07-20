import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import csv

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

N_train=12000
N_test=3000
N_input=200
N_features=20
N_hidden_layer=2

SNR=30

b_min, b_max = 10.0, 2000.0 # s/mm^2
b=torch.linspace(b_min,b_max,N_input)
#b=(b-torch.min(b))/(torch.max(b)-torch.min(b))

S0_min, S0_max = 0.5, 5.0
S= S0_min + (S0_max - S0_min) * torch.rand(N_train,1)
#S=(S-torch.min(S))/(torch.max(S)-torch.min(S))

D_min, D_max  = 0.1e-3, 3.0e-3
D=D_min + (D_max - D_min) * torch.rand(N_train,1)
#D=(D-torch.min(D))/(torch.max(D)-torch.min(D))

X=torch.zeros(N_train,N_input)
for i in range(0,N_train):
    X[i,:]=S[i]*torch.exp(-b*D[i])
sigma = torch.mean(S) / SNR
noise = torch.normal(0, sigma, size=(N_train, N_input))
X=X+noise

Y=torch.ones(N_train,2)
Y[:,0]=torch.transpose(S,0,1)
Y[:,1]=torch.transpose(D,0,1)

training_data=Signals(X,Y)

S0_min, S0_max = 0.5, 5.0
S= S0_min + (S0_max - S0_min) * torch.rand(N_test,1)
#S=(S-torch.min(S))/(torch.max(S)-torch.min(S))

D_min, D_max  = 0.1e-3, 3.0e-3
D=D_min + (D_max - D_min) * torch.rand(N_test,1)
#D=(D-torch.min(D))/(torch.max(D)-torch.min(D))

X=torch.zeros(N_test,N_input)
for i in range(0,N_test):
    X[i,:]=S[i]*torch.exp(-b*D[i])
sigma = torch.mean(S) / SNR
noise = torch.normal(0, sigma, size=(N_test, N_input))
X=X+noise

Y=torch.ones(N_test,2)

Y[:,0]=torch.transpose(S,0,1)
Y[:,1]=torch.transpose(D,0,1)

testing_data=Signals(X,Y)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

b=b.to(device)

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
        x=F.linear(x,selector)
        outputs= {"latent": x, "reg": reg, "idx": torch.argmax(selector, dim=1)}
        return outputs


class Normalization(nn.Module):
    def __init__(self,input_dim, features, output_dim=2):
        super().__init__()
        self.nparam = output_dim # Number of parameters
        normlist = []
        for pp in range(output_dim):
            normlist.append(nn.Linear(1,1, bias=False))
        self.sgmnorm = nn.ModuleList(normlist)
        self.param_min = torch.tensor([S0_min, D_min], device=device)
        self.param_max = torch.tensor([S0_max, D_max], device=device)
        self.param_name = ['S0', 'D']
        self.con_one=torch.tensor([1.0],device=device)
        self.con_two=torch.tensor([2.0],device=device)
    
    def getnorm(self,x):
        if x.dim()==1:
            normt = torch.zeros(self.nparam, device=device)
            for pp in range(self.nparam):
                bt = torch.zeros(self.nparam, device=device)
                bt[pp]=1.0
                con_one = torch.tensor([1.0])
                bt = self.sgmnorm[pp](con_one)*bt
                normt = normt + bt
            normt = torch.abs(normt)
            x=x*normt
        elif x.dim()==2:
            normt = torch.zeros(x.shape[0],self.nparam, device=device)
            for pp in range(self.nparam):
                bt = torch.zeros(x.shape[0],self.nparam,device=device)
                bt[:,pp] = 1.0
                bt = self.sgmnorm[pp](self.con_one).detach().clone() * bt
                normt = normt + bt
            normt = torch.abs(normt)
            x = x*normt
        else:
            raise RuntimeError('getnorm() only accepts 1D or 2D inputs')
            
        return x
    
    def getparams(self,x):
        x = torch.log(x)
        x = x - torch.log(torch.log(self.con_two))
        x = self.getnorm(x)
        x = torch.sigmoid(x)
        if x.dim()==1:
            x = (self.param_max - self.param_min)*x + self.param_min
                
        elif x.dim()==2:
            t_ones = torch.ones(x.shape[0],1, device=device)
            max_val = torch.cat( ( self.param_max[0]*t_ones , self.param_max[1]*t_ones), 1  )
            min_val = torch.cat( ( self.param_min[0]*t_ones , self.param_min[1]*t_ones), 1   )
            x = (max_val - min_val)*x + min_val
        return x

    def getsignals(self,x):
        if x.dim()==1:
            b_D = b*x[1]
            s_tot=x[0]*torch.exp(-b_D)
        elif x.dim()==2:
            Nvox=x.shape[0]
            s_tot = torch.zeros(Nvox,N_input, device=device)
            s_tot =x[:,0:1]*torch.exp(-b.unsqueeze(0)*x[:,1:2])
        return s_tot

    def forward(self, x):
        param=self.getparams(x)
        signal=self.getsignals(param)
        outputs= {"parameters": param, "signal": signal}
        return outputs

class CAE(nn.Module):
    def __init__(self, input_dim=N_input, features=N_features, n_hidden_layers=N_hidden_layer, dropout=0.0):
        super().__init__()
        indices2=np.arange(2+n_hidden_layers)
        data_indices2=np.array([indices2[0], indices2[-1]])
        data2=np.array([features,2])
        layer_sizes=np.interp(indices2, data_indices2, data2).astype(int)
        n_layers=len(layer_sizes)
        layers=[]
        for i in range(1, n_layers):
            if i==n_layers-1:
                layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
                layers.append(nn.Softplus())
            else:
                layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
                layers.append(nn.ReLU(True))
            
        print(layer_sizes,layers)
        self.encoder=ConcreteLayer(input_dim, features)
        self.decoder=nn.Sequential(*layers)
        self.normalization=Normalization(input_dim, features)

    def forward(self, x, temperature, random, threshold):
        outputs=self.encoder(x, random, temperature, threshold)
        x=self.decoder(outputs["latent"])
        x=self.normalization(x)
        reg=outputs["reg"]
        returns = {'Parameters': x['parameters'], 'MRI': x['signal'], 'REG': reg, 'Idx': outputs["idx"]}
        return returns

def temp_value(num_epochs, temp_base, temp_min, epoch):
    temp=temp_base*(temp_min/temp_base)**(epoch/num_epochs)
    return temp 
    
model=CAE().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 50

temp_base=10
temp_min=0.1
threshold=1
strength=0.1

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(epoch, model, train_loader, optimizer):
    model.train()
    sum_loss=0
    num_batches=len(train_loader)
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        temp=temp_value(epochs, temp_base, temp_min, epoch)
        optimizer.zero_grad()
        returns = model(X, temp, True, threshold)
        reg=returns['REG']
        recon_batch=returns['Parameters']
        loss = loss_function(recon_batch, y)+strength*reg
        sum_loss+=loss
        loss.backward()
        optimizer.step()
    sum_loss/=num_batches
    if epoch%5 ==0:
        print(f"Epoch {epoch}, Average Loss: {sum_loss:.6f}")

def test_loop(epoch, dataloader, model, loss_fn, loss_threshold, indices):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, Scorrect, Dcorrect = 0, 0, 0
    S_absolute_errors = []
    D_absolute_errors = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            temp = temp_value(epochs, temp_base, temp_min, epoch)
            returns = model(X, temp, False, threshold)
            pred = returns['Parameters']
            idx = returns['Idx']
            test_loss += loss_fn(pred, y).item()
            # S_m = y[:, 0] - y[:, 0] * 0.1
            # S_p = y[:, 0] + y[:, 0] * 0.1
            # D_m = y[:, 1] - y[:, 1] * 0.1
            # D_p = y[:, 1] + y[:, 1] * 0.1
            for i in range(len(pred)):
                # Absolute errors for S and D
                S_error = abs(pred[i, 0] - y[i, 0])
                D_error = abs(pred[i, 1] - y[i, 1])
                S_absolute_errors.append(S_error.item())
                D_absolute_errors.append(D_error.item())
                # if pred[i][0] > S_m[i] and pred[i][0] < S_p[i]:
                #     Scorrect += 1
                # if pred[i][1] > D_m[i] and pred[i][1] < D_p[i]:
                #     Dcorrect += 1
    test_loss /= num_batches
    # Scorrect /= size
    # Dcorrect /= size
    # Calculate mean, max, min absolute errors
    S_mean_ae = sum(S_absolute_errors) / len(S_absolute_errors)
    S_max_ae = max(S_absolute_errors)
    S_min_ae = min(S_absolute_errors)

    D_mean_ae = sum(D_absolute_errors) / len(D_absolute_errors)
    D_max_ae = max(D_absolute_errors)
    D_min_ae = min(D_absolute_errors)
    if test_loss<loss_threshold:
        indices=idx.tolist().copy()
        loss_threshold=test_loss
    if epoch %5 == 0:
        print(
            f"Test Error: \n"
            #f"Accuracy on S: {(100*Scorrect):>0.1f}%, Accuracy on D: {(100*Dcorrect):>0.1f}%"
            f"Avg loss: {test_loss:>8f} \n"
            f"S Absolute Error - Mean: {S_mean_ae:.4f}, Max: {S_max_ae:.4f}, Min: {S_min_ae:.4f}\n"
            f"D Absolute Error - Mean: {D_mean_ae:.4f}, Max: {D_max_ae:.4f}, Min: {D_min_ae:.4f}\n"
        )
    return indices, loss_threshold  # Retourne les valeurs mises à jour

# Boucle principale
indices = None
loss_threshold = float('inf')  # Initialisation

for epoch in range(1, epochs + 1):
    train_loop(epoch, model, train_dataloader, optimizer)
    indices, loss_threshold = test_loop(epoch, test_dataloader, model, loss_function, loss_threshold,indices)

chemin = "./subset_param.csv"

with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

    mon_fichier_ecrire.writerow(indices)