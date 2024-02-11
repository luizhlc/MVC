from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset
import pandas as pd

class CustomDataSet(Dataset):
    def __init__(self, csv_file):
        self.classes_k = {'Classe A':0,'Classe B':1, 'Classe C':2,'Classe D':3,'Classe E':4}
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        label = self.classes_k[self.df.loc[index]['Classe']]
        data = torch.tensor(self.df.loc[index][1:])
        return data, label


class MVC(nn.Module):
    def __init__(self, in_features, out_features=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        result = self.layers(x).argmax(1).data.cpu().numpy()[0]
        classes_k = ['Classe A', 'Classe B', 'Classe C', 'Classe D', 'Classe E']
        return classes_k[result]
    
def training_epoch(net, optimizer, loss_f, dataloader):
    t_loss = 0
    t_acc = 0
    net.train()
    for batch_idx, (data,label) in enumerate(dataloader):
        optimizer.zero_grad()
        output = net(data)
        loss = loss_f(output,label)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * data.size(0)
        predicted_class = output.argmax(1)
        correct_ones = (predicted_class == label).type(torch.float)
        t_acc += correct_ones.sum().item()
    return t_loss, t_acc

def validation_epoch(net, loss_f, dataloader):
    v_loss = 0
    v_acc = 0
    net.eval()
    for data,label in dataloader:
        output = net(data)
        loss = loss_f(output,label)
        v_loss += loss.item() * data.size(0)
        predicted_class = output.argmax(1)
        correct_ones = (predicted_class == label).type(torch.float)
        v_acc += correct_ones.sum().item()
    return v_loss, v_acc

def save_model(model_p, epoch, model, optimizer, v_loss, isBest):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': v_loss,
        }, f"{model_p}/model_{epoch}.checkpoint")
    if isBest:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': v_loss,
        }, f"{model_p}/model_best.checkpoint")


