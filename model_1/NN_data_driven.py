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
N_input=512

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

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        self.leaky=nn.LeakyReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.relu(self.fc2(h1))

    def decode(self, z):
        h2 = self.leaky(self.fc3(z))
        return self.leaky(self.fc4(h2))

    def forward(self, x):
        q = self.encode(x.view(-1, 512))
        return self.decode(q)

model=DAE()

learning_rate = 1e-3
batch_size = 64
epochs = 20

def train(epoch, model, train_loader, optimizer, cuda=True):
    model.train()
    sum_loss=0
    num_batches=len(train_loader)
    for batch, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        recon_batch = model(X)
        loss = loss_function(recon_batch, y)
        sum_loss+=loss
        loss.backward()
        optimizer.step()
        #print(f"Loss: {loss.item():.6f}")
    sum_loss/=num_batches
    print(f"Epoch{epoch}, Average loss: {sum_loss}")
    

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct =0, 0
    absolute_errors=[]
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            y_m=y-y*0.1
            y_p=y+y*0.1
            test_loss += loss_fn(pred,y).item()
            for i in range(len(pred)):
                for j in range(512):
                    error = abs(pred[i, j] - y[i, j])
                    absolute_errors.append(error.item())
                    if pred[i][j] > y_m[i][j] and pred[i][j] < y_p[i][j]:
                        correct+= 1
    test_loss/=num_batches
    correct/=(size*512)
    mean_ae = sum(absolute_errors) / len(absolute_errors)
    max_ae = max(absolute_errors)
    min_ae = min(absolute_errors)
    if epoch%5==0:
        plt.figure()
        plt.plot(b,y[0],'ro-',label='expectation')
        plt.plot(b,pred[0],'bo',label='prediction')
        plt.ylabel('Signal')
        plt.xlabel('b values')
        plt.legend()
        plt.grid()
        plt.savefig(f"dd_epoch_{epoch}_2")
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