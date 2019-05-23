# MNIST contains handwritten digit and
# is trained to recognize the handwritten digits

# Importing libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Device configuration
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # used to run on GPU

#Hyper parameters
input_size=784 # image size (total 784 pixels)
hidden_size=500 # hidden layer 1
hidden_size_2=100 # hidden layer 2
num_classes=10 # output layer (for digits 0 ot 9)
num_epochs=5 #Number of epochs is the number of times the whole
             # training data is shown to the network while training
batch_size=100
learning_rate=0.001

# MNIST Dataset
train_dataset=torchvision.datasets.MNIST(root='../../data',train=True,
                                         transform=transforms.ToTensor(),
                                         download=True) # the MNIST dataset is downloaded and train function selected as passed by train=True

test_dataset=torchvision.datasets.MNIST(root='../../data',train=False,
                                         transform=transforms.ToTensor()) # test function selected from MNIST dataset (train=False)
                                        # tranform to tensor tranforms
                                        # image dataset into tensor form
# Data Loader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_size_2,num_classes):
        super(NeuralNet, self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size) # for passing from layer of bipartite graph of 784 elements into part of 500 elements
        self.relu=nn.ReLU() # to perform relu function on first layer (784)
        self.fc2=nn.Linear(hidden_size,hidden_size_2) # convert 500 elements into 100 elements (next layer size)
        self.fc3=nn.Linear(hidden_size_2,num_classes) # convert 500 elements into 10 elements (digits 0 to 9)

    def forward(self,x): # Forward propogation
        out=self.fc1(x) # feeding image tensor into first layer
        out=self.relu(out) # converting first layer input into relu function
        out=self.fc2(out) # propogating hidden layer to 2nd hidden layer (100)
        out=self.fc3(out) # propogating 2nd hidden layer to ouptput layer (10)
        return out

model=NeuralNet(input_size,hidden_size,hidden_size_2,num_classes).to(device)

# Loss and optimizer

criterion=nn.CrossEntropyLoss()
# optimizing the learning rate
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

# Train the model
total_step=len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images=images.reshape(-1,28*28).to(device) # 28*28=784
        labels=labels.to(device)

        # Forward pass
        outputs=model(images) # passing into Neural Net class
        loss=criterion(outputs,labels) #checking for cross entropy loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print('Epoch [{}/{}], step[{}/{}], Loss: {:.4f}'
            .format(epoch+1,num_epochs,i+1,total_step,loss.item()))
            # prints epoch i/total and step j/total and loss

# Test the model
# In test phase, we do not need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100*correct/total))

    #Save the model and checkpoint
    torch.save(model.state_dict(),'model.ckpt')

# Testing on a random image
img=Image.open('image.jpg')
#img=torchvision.transforms.functional.to_grayscale(img)
img.show()
transform=transforms.ToTensor()
img=transform(img)
img.resize_((1,28*28))
model = NeuralNet(input_size,hidden_size,hidden_size_2,num_classes)
trained_model=torch.load('model.ckpt')
model.load_state_dict(trained_model)
outputs=model(img)
_,ans=torch.max(outputs.data,1)
print(ans)
