import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

N_train=1000
N_test=192
N_input=500

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
#X_norm=X/X[:,0:1]
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
#X_norm=X/X[:,0:1]
sigma = torch.mean(S) / SNR
noise = torch.normal(0, sigma, size=(N_test, N_input))
X=X+noise

testing_data=Signals(X,X)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=False)

class Normalization(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.nparam = output_dim # Number of parameters
        normlist = []
        for pp in range(output_dim):
            normlist.append(nn.Linear(1,1, bias=False))
        self.sgmnorm = nn.ModuleList(normlist)
        self.param_min = torch.tensor([S0_min, D_min])
        self.param_max = torch.tensor([S0_max, D_max])
        self.param_name = ['S0', 'D']
        self.con_one=torch.tensor([1.0])
        self.con_two=torch.tensor([2.0])
    
    def getnorm(self,x):
        if x.dim()==1:
            normt = torch.zeros(self.nparam)
            for pp in range(self.nparam):
                bt = torch.zeros(self.nparam)
                bt[pp]=1.0
                con_one = torch.tensor([1.0])
                bt = self.sgmnorm[pp](con_one)*bt
                normt = normt + bt
            normt = torch.abs(normt)
            x=x*normt
        elif x.dim()==2:
            normt = torch.zeros(x.shape[0],self.nparam)
            for pp in range(self.nparam):
                bt = torch.zeros(x.shape[0],self.nparam)
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
            t_ones = torch.ones(x.shape[0],1)
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
            s_tot = torch.zeros(Nvox,N_input)
            s_tot =x[:,0:1]*torch.exp(-b.unsqueeze(0)*x[:,1:2])
        return s_tot

    def forward(self, x):
        param=self.getparams(x)
        signal=self.getsignals(param)
        outputs= {"parameters": param, "signal": signal}
        return outputs

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(500, 335)
        self.fc2 = nn.Linear(335, 171)
        self.fc3 = nn.Linear(171, 2)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.normalization=Normalization()

    def forward(self, x):
        x = self.relu(self.fc1(x.view(-1, 500)))
        x = self.relu(self.fc2(x))
        x = self.softplus(self.fc3(x))
        q = self.normalization(x)
        returns = {'Parameters': q['parameters'], 'MRI': q['signal']}
        return returns

# def equation(recon_batch):
#     size=len(recon_batch)
#     recon_eq=torch.zeros(size,N_input)
#     # amplitude=torch.pow((torch.pow(recon_batch[:,0],2)/SNR),0.5)
#     # noise=torch.normal(0,amplitude)
#     for i in range(size):
#         recon_eq[i] = recon_batch[i,0]*torch.exp(-b*recon_batch[i,1])#+noise[i]
#     return recon_eq


model=DAE()

learning_rate = 1e-3
batch_size = 64
epochs = 30

def train(epoch, model, train_loader, optimizer, cuda=True):
    model.train()
    sum_loss=0
    num_batches=len(train_loader)
    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch = model(X)
        recon_eq = recon_batch['MRI']
        loss = loss_function(recon_eq, y)
        sum_loss+=loss
        loss.backward()
        optimizer.step()
    sum_loss/=num_batches
    print(f"Epoch {epoch}, Average loss: {sum_loss:.6f}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct =0, 0
    absolute_errors=[]
    with torch.no_grad():
        for X,y in dataloader:
            recon_batch = model(X)
            pred = recon_batch['MRI']
            y_m=y-y*0.1
            y_p=y+y*0.1
            test_loss += loss_fn(pred,y).item()
            for i in range(len(pred)):
                for j in range(500):
                    error = abs(pred[i, j] - y[i, j])
                    absolute_errors.append(error.item())
                    if pred[i][j] > y_m[i][j] and pred[i][j] < y_p[i][j]:
                        correct+= 1
    test_loss/=num_batches
    correct/=(size*N_input)
    mean_ae = sum(absolute_errors) / len(absolute_errors)
    max_ae = max(absolute_errors)
    min_ae = min(absolute_errors)
    if epoch%5==0:
        plt.figure()
        plt.plot(b,pred[0],'bo',label='prediction')
        plt.plot(b,y[0],'ro-',label='expectation')
        plt.ylabel('Signal')
        plt.xlabel('b values')
        plt.legend()
        plt.grid()
        plt.savefig(f"epoch_{epoch}")
        print(
            f"Test Error: \n"
            f"Accuracy : {(100*correct):>0.1f}%"
            f"Avg loss: {test_loss:>8f} \n"
            f"Absolute Error - Mean: {mean_ae:.4f}, Max: {max_ae:.4f}, Min: {min_ae:.4f}\n"
        )

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    train(epoch, model, train_dataloader, optimizer, True)
    test_loop(test_dataloader, model, loss_function)