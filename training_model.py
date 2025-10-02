
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))

])

#datasets
train_dataset = datasets.MNIST(root = './data', train = True, transform= transformer, download= True)
test_dataset = datasets.MNIST( root = './data', train = True , transform = transformer , download =  True)

#dataloaders
train_dataloader = DataLoader(train_dataset, batch_size = 32 , shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = 32 , shuffle = True)

class Model(nn.Module):
  def __init__(self):
        super().__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(1,32,2,padding= 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride= 2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(32,64,2,padding= 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride= 2),
            nn.Dropout2d(p=0.3),
        )



        self.classifer = nn.Sequential(
            nn.Linear(64*9*9, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64,10)
        )
  def forward(self, data):
      data = self.convlayer(data)
      data = data.view(data.size(0),-1)
      return self.classifer(data)

epochs = 90
lr = 0.005
model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr = lr ,  weight_decay = 1e-5)
loss_fun = nn.CrossEntropyLoss()
a = []
for epoch in range(epochs):

  total_loss = 0
  model.train()

  for images , labels in train_dataloader:

    images,labels = images.to(device), labels.to(device)
    pred = model(images)
    loss = loss_fun(pred,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(epoch,'--- Loss',total_loss/32)
  a.append(total_loss/32)

score = cross_val_score(model,train_dataset,)
plt.plot(a)
plt.show()
torch.save(model.state_dict(),'proj1.pth')