import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import numpy as np
# import matplotlib.pyplot as plt

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

training_data=Signals(X,X)

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

testing_data=Signals(X,X)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

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

class CAE(nn.Module):
    def __init__(self, input_dim=N_input, features=N_features, n_hidden_layers=N_hidden_layer, dropout=0.0):
        super().__init__()
        indices2=np.arange(2+n_hidden_layers)
        data_indices2=np.array([indices2[0], indices2[-1]])
        data2=np.array([features,N_input])
        layer_sizes=np.interp(indices2, data_indices2, data2).astype(int)
        n_layers=len(layer_sizes)
        layers=[]
        for i in range(1, n_layers):
            if i==n_layers-1:
                layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
            else: 
                layers.append(nn.Linear(layer_sizes[i-1],layer_sizes[i]))
                layers.append(nn.LeakyReLU(True))
            
        print(layer_sizes,layers)
        self.encoder=ConcreteLayer(input_dim, features)
        # self.decoder=nn.Sequential(
        #     nn.Linear(features, 116),
        #     nn.LeakyReLU(),
        #     nn.Linear(116, 158),
        #     nn.LeakyReLU(),
        #     nn.Linear(158,input_dim)
        # )
        self.decoder=nn.Sequential(*layers)

    def forward(self, x, temperature, random, threshold):
        outputs=self.encoder(x, temperature, random, threshold)
        x=self.decoder(outputs["latent"])
        reg=outputs["reg"]
        returns = {'X_rec': x, 'REG': reg, 'Idx': outputs["idx"]}
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
        returns = model(X, True, temp, threshold)
        reg=returns['REG']
        recon_batch=returns['X_rec']
        loss = loss_function(recon_batch, y)+strength*reg
        sum_loss+=loss
        loss.backward()
        optimizer.step()
    sum_loss/=num_batches
    if epoch%5==0:
        print(f"Epoch {epoch}, Average Loss: {sum_loss:.6f}")

def test_loop(epoch, dataloader, model, loss_fn,loss_threshold,indices):
    model.eval()
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct =0, 0
    absolute_errors=[]
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            temp=temp_value(epochs, temp_base, temp_min, epoch)
            returns = model(X, False, temp, threshold)
            pred =returns['X_rec']
            idx = returns['Idx']
            y_m=y-y*0.1
            y_p=y+y*0.1
            test_loss += loss_fn(pred,y).item()
            for i in range(len(pred)):
                for j in range(N_input):
                    error = abs(pred[i, j] - y[i, j])
                    absolute_errors.append(error.item())
                    if pred[i][j] > y_m[i][j] and pred[i][j] < y_p[i][j]:
                        correct+= 1
    test_loss/=num_batches
    correct/=(size*N_input)
    mean_ae = sum(absolute_errors) / len(absolute_errors)
    max_ae = max(absolute_errors)
    min_ae = min(absolute_errors)
    if test_loss<loss_threshold:
        indices=idx.tolist().copy()
        loss_threshold=test_loss
    if epoch%5==0:
        print(
                f"Test Error: \n"
                f"Accuracy: {(100*correct):>0.1f}% \n"
                f"Avg loss: {test_loss:>8f} \n"
                f"Absolute Error - Mean: {mean_ae:.4f}, Max: {max_ae:.4f}, Min: {min_ae:.4f}\n"
            )
    return indices, loss_threshold  # Retourne les valeurs mises à jour

# Boucle principale
indices = None
loss_threshold = float('inf')  # Initialisation

for epoch in range(1, epochs + 1):
    train_loop(epoch, model, train_dataloader, optimizer)
    indices, loss_threshold = test_loop(epoch, test_dataloader, model, loss_function,loss_threshold,indices)

chemin = "./subset_data.csv"

with open(chemin, mode='w') as mon_fichier:
    mon_fichier_ecrire = csv.writer(mon_fichier, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

    mon_fichier_ecrire.writerow(indices)