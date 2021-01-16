import os
from torchvision import datasets
import dataiter
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms as T
import scipy
import torch
import torchvision.models as models
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from glob import glob
use_cuda = torch.cuda.is_available()
#use_cuda=False
torch.cuda.empty_cache()
# calculate time
import time
start=time.time()

batch_size =10
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
data_dir = 'birdImages/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')
valid_dir= os.path.join(data_dir, 'valid/')
#data_transform = transforms.Compose([transforms.RandomResizedCrop(64),
#                                      transforms.ToTensor()])
#data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.ToTensor()])

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
#    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

num_workers=0

# prepare data loaders




train_data = datasets.ImageFolder(train_dir,transform=data_transform)
train = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=0)

valid_data = datasets.ImageFolder(valid_dir,transform=data_transform)
valid = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=0)

test_data = datasets.ImageFolder(test_dir,transform=data_transform)
test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,  num_workers=0)


loaders_scratch = {"train":train,"valid":valid,"test":test}
# print out some data stats
print('Num training images: ', len(train_data))
print('Num valid images: ', len(valid_data))
print('Num test images: ', len(test_data))
print("train loader size",len(loaders_scratch))


# obtain one batch of training images
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
dataiter = iter(train)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
#display 20 images


#for idx in np.arange(batch_size):
#    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#    imshow(images[idx])
#    ax.set_title(labels[idx])
#CNN
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import Linear,ReLU,CrossEntropyLoss,Sequential,Conv2d,MaxPool2d,Module,Softmax,BatchNorm2d,Dropout
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.empty_cache()
# define the CNN architecture
# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        ##self.mp=nn.MaxPool2d(1, stride=2)
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3,32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.1),

            #nn.ReflectionPad2d(1),
            #nn.Conv2d(32, 32, kernel_size=3),
            #nn.Sigmoid(),
            #nn.BatchNorm2d(32),
            #nn.Dropout2d(p=.2),

            #nn.ReflectionPad2d(1),
            #nn.Conv2d(32, 16, kernel_size=3),
            #nn.Sigmoid(),
            #nn.BatchNorm2d(16),
            #nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.1),
        )
        self.Linear_layer=Sequential(

            nn.Linear(16*224*224,150),
            nn.ReLU(),
            nn.Dropout2d(p=.1),



            nn.Linear(150,133),

             )


    def forward(self, x):
        ## Define forward behavior


        x=self.cnn1(x)


        x=x.view(x.shape[0],-1)


        x=self.Linear_layer(x)

        #print ("x is",x)
        return x



#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()
summary(model_scratch
     )
# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


# optimizer
import torch.optim as optim
from torch.optim import Adam
### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.0003, momentum=0.9)
#optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.0002)

# Train
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    correct=0
    if use_cuda:
        print( " USING GPU")
    else:
        print ("CPU")
    optimizer.zero_grad()
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0
        valid_loss = 0
        i=0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):

        #for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            #if i%400==0:
            #    print(i," batches done ",int(time.time()-start),"s")
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # perform a single optimization step (parameter update)

            #print (loss)
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (float(loss.data) - train_loss))
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            i += 1
            del loss, output


        #print("correct", correct," out of",i)

        correct=0
        i=0
        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
        #for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
                # calculate the batch loss
            loss = criterion(output, target)
                # update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (float(loss.data) - valid_loss))



            ## update the average validation loss
        #train_loss = train_loss/len(loaders['train'].dataset)
        #valid_loss = valid_loss/len(loaders['valid'].dataset)


        # print training/validation statistics
        if valid_loss <= valid_loss_min +99:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

        ## TODO: save the model if validation loss has decreased

        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # return trained model
    return model


# train the model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch,
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))

# TEST
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)



        # calculate the loss
        loss = criterion(output, target)

        #print (target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        #print (pred)
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        #print ("out",pred,"targ",target)
        #print (correct)
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


# print run time
print ("run time",int(time.time()-start),"s")
