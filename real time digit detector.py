
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))

])



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
  
model = Model()
state_dict = torch.load('C:/Users/nisha/OneDrive - Indian Institute of Technology Guwahati/Documents/projects/pytorch/proj1_.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
old_param = [ param.clone().detach() for param in model.parameters()]




#initializing the canvas

canvas = np.zeros((280,280), dtype = np.uint8)*255
drawing = False
def draw(event, x , y , flags , param):

  global drawing
  if event == cv2.EVENT_LBUTTONDOWN :
    drawing = True

  elif event == cv2.EVENT_MOUSEMOVE and drawing :
    cv2.circle(canvas,(x,y),10,(120),-1)

  elif event == cv2.EVENT_LBUTTONUP :
    drawing = False



cv2.namedWindow("press p to predict c to clear and q to quit")
cv2.setMouseCallback("press p to predict c to clear and q to quit",draw)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005 , weight_decay = 1e-5)

while True:
    cv2.imshow("press p to predict c to clear and q to quit",canvas)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('p') :
        img = cv2.resize(canvas, (28, 28))
        img = cv2.bitwise_not(img)  # Invert so the digit is dark on white
        img = img.astype(np.uint8)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.unsqueeze(0).unsqueeze(0)
        img = transforms.Normalize((0.1307,), (0.3081,))(img)
        

        with torch.no_grad():
            output = model(img.to(device))
            print(output)
            pred = output.argmax(dim=1).item()
        print('the digit is',pred)
        cv2.putText(canvas, f"pred: {pred}", (10,270),cv2.FONT_HERSHEY_SIMPLEX,1.0,200,2)
        feedback = input("is predicted value right ?(y/n)")

        if feedback == 'y':
          label = pred
        
        else:
          label = int(input('Right value: '))
        
        label = torch.tensor([label],dtype = torch.long).to(device)

        n_out = model(img.to(device))
        
        model.eval()
        label = label.to(device)
        loss = loss_fn(n_out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updated = False
        for old,new in zip(old_param,model.parameters()):
           if not torch.equal(old,new):
              updated = True
              break
        
        print('updated' if updated else 'not updated')

        

    elif key == ord('c'):
        canvas[:] = 255

    elif key == ord ('q') :
        break

torch.save(model.state_dict(), 'Downloads/proj_1.pth')
cv2.destroyAllWindows()