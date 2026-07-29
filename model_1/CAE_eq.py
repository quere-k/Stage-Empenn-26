import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import csv

#Class to create datasets
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

#Hyperparameters initialization
N_train=12000
N_test=3000
N_input=200
N_features=20
N_hidden_layer=2

SNR=30

#Generation of the datasets
#Range of parameters
b_min, b_max = 10.0, 2000.0 # s/mm^2
b=torch.linspace(b_min,b_max,N_input)#evenly spaced

S0_min, S0_max = 0.5, 5.0
S= S0_min + (S0_max - S0_min) * torch.rand(N_train,1)#randomly generated

D_min, D_max  = 0.1e-3, 3.0e-3
D=D_min + (D_max - D_min) * torch.rand(N_train,1)#randomly generated

#Signal matrix N_train*N_input
X=torch.zeros(N_train,N_input)
for i in range(0,N_train):
    X[i,:]=S[i]*torch.exp(-b*D[i])
sigma = torch.mean(S) / SNR
noise = torch.normal(0, sigma, size=(N_train, N_input))
X=X+noise

training_data=Signals(X,X)

S0_min, S0_max = 0.5, 5.0
S= S0_min + (S0_max - S0_min) * torch.rand(N_test,1)

D_min, D_max  = 0.1e-3, 3.0e-3
D=D_min + (D_max - D_min) * torch.rand(N_test,1)

X=torch.zeros(N_test,N_input)
for i in range(0,N_test):
    X[i,:]=S[i]*torch.exp(-b*D[i])
sigma = torch.mean(S) / SNR
noise = torch.normal(0, sigma, size=(N_test, N_input))
X=X+noise

testing_data=Signals(X,X)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

#Use CPU when available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

b=b.to(device)
#Class for the construction of the concrete selection layer
class ConcreteLayer(nn.Module):
    def __init__(self, num_inputs, num_features, pi_dropout=0.0):
        super().__init__()
        #Store parameters
        self.num_inputs=num_inputs
        self.num_features=num_features

        #Initialization
        logits_init=torch.rand(num_features,num_inputs)
        logits_init=logits_init/torch.sum(logits_init)
        self.logits=nn.Parameter(logits_init, requires_grad=True) #make logits learnable for the model
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
    
    def regularization(self, logits, threshold): # Regularization function (avoid multiple selection)
        num_inputs=self.num_inputs
        pi=F.softmax(logits, dim=1)
        L=torch.zeros(num_inputs)
        for i in range(num_inputs):
            L[i]=F.relu(torch.sum(pi[:,i]-threshold)) #Probability sum for a same input over selection neurons
        reg=torch.sum(L)
        return reg

    def forward(self, x, random, temperature, threshold, hard=False):
        selector,reg=self.sample_matrix(temperature, random, threshold, hard)
        x=F.linear(x,selector)
        outputs= {"latent": x, "reg": reg, "idx": torch.argmax(selector, dim=1)}
        return outputs

#Class for the normalization
class Normalization(nn.Module):
    def __init__(self,input_dim, features, output_dim=2):
        super().__init__()
        self.nparam = output_dim # Number of parameters
        normlist = []
        for pp in range(output_dim):
            normlist.append(nn.Linear(1,1, bias=False))
        self.sgmnorm = nn.ModuleList(normlist)
        self.param_min = torch.tensor([S0_min, D_min], device=device) #Minimum values
        self.param_max = torch.tensor([S0_max, D_max], device=device) #Maximum values
        self.param_name = ['S0', 'D']
        self.con_one=torch.tensor([1.0],device=device)
        self.con_two=torch.tensor([2.0],device=device)
    
    def getnorm(self,x): # learn the normalization factors
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
    
    def getparams(self,x): # calculate the value of parameters using normalization factors and min and max values
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

    def getsignals(self,x): # calculate signals using parameters values
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

# Class for the construction of the concrete auto-encoder = selection layer + decoder
class CAE(nn.Module):
    def __init__(self, input_dim=N_input, features=N_features, n_hidden_layers=N_hidden_layer, dropout=0.0):
        super().__init__()
        indices2=np.arange(2+n_hidden_layers)
        data_indices2=np.array([indices2[0], indices2[-1]])
        data2=np.array([features,2])
        layer_sizes=np.interp(indices2, data_indices2, data2).astype(int) # 1D linear interpolation for hidden neurons
        n_layers=len(layer_sizes)
        layers=[]
        for i in range(1, n_layers): #Contruction of the hidden layers
            layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
            layers.append(nn.ReLU(True))
            if i==n_layers-1:
                layers.append(nn.Softplus())
        print(layer_sizes)
        self.encoder=ConcreteLayer(input_dim, features)
        self.decoder=nn.Sequential(*layers)
        self.normalization=Normalization(input_dim, features)

    def forward(self, x, temperature, random, threshold):
        outputs=self.encoder(x, random, temperature, threshold) #concrete selection layer
        x=self.decoder(outputs["latent"]) #decoder
        x=self.normalization(x)
        reg=outputs["reg"]
        returns = {'Parameters': x['parameters'], 'MRI': x['signal'], 'REG': reg, 'Idx': outputs["idx"]}
        return returns

def temp_value(num_epochs, temp_base, temp_min, epoch): # Exponential decrease for the temperature
    temp=temp_base*(temp_min/temp_base)**(epoch/num_epochs)
    return temp 

model=CAE().to(device)

#Hyperparameters for the training + loss calculation
learning_rate = 1e-3
batch_size = 64
epochs = 50

temp_base=10
temp_min=0.1
threshold=1
strength=0.1

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training loop
def train_loop(epoch, model, train_loader, optimizer):
    model.train()
    sum_loss=0
    num_batches=len(train_loader)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        temp=temp_value(epochs, temp_base, temp_min, epoch) #temperature update
        optimizer.zero_grad()
        returns = model(X, temp, True, threshold) #training of the model
        reg=returns['REG']
        recon_eq = returns['MRI']
        loss = loss_function(recon_eq, y)+strength*reg #calculation of the loss
        sum_loss+=loss
        loss.backward()
        optimizer.step()
    sum_loss/=num_batches
    if epoch%5 == 0:
        print(f"Epoch {epoch}, Average Loss: {sum_loss:.6f}")

#Testing loop
def test_loop(epoch, dataloader, model, loss_fn, loss_threshold,indices):
    model.eval()
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss = 0
    absolute_errors=[]
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            temp=temp_value(epochs, temp_base, temp_min, epoch) #temperature update
            returns = model(X, temp, False, threshold) #testing of the model
            pred = returns['MRI']
            idx = returns['Idx']
            test_loss += loss_fn(pred,y).item()
            for i in range(len(pred)):
                for j in range(N_input):
                    error = abs(pred[i, j] - y[i, j]) #calculation of the error
                    absolute_errors.append(error.item())
    test_loss/=num_batches
    mean_ae = sum(absolute_errors) / len(absolute_errors)
    max_ae = max(absolute_errors)
    min_ae = min(absolute_errors)
    if test_loss<loss_threshold:
        indices=idx.tolist().copy()
        loss_threshold=test_loss
    if epoch%5==0:
        print(
            f"Test Error: \n"
            f"Avg loss: {test_loss:>8f} \n"
            f"Absolute Error - Mean: {mean_ae:.4f}, Max: {max_ae:.4f}, Min: {min_ae:.4f}\n"
        )
    return indices, loss_threshold  # return the indices and the new threshold

indices = None
loss_threshold = float('inf')  # Initialisation

#main loop
for epoch in range(1, epochs + 1):
    train_loop(epoch, model, train_dataloader, optimizer)
    indices, loss_threshold = test_loop(epoch, test_dataloader, model, loss_function, loss_threshold,indices)

#store the best subset in a csv file 
chemin = "./subset_eq.csv"

with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

    mon_fichier_ecrire.writerow(indices)