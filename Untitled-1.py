import torch
import numpy as np
from torchvision import datasets , models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("Cuda is not available, training on CPU...")
else:
    print("Cuda is available, training on GPU...")

train_or_not = input("Do you want to retrain the model?! Yes(y) , No(n) \n")
if train_or_not == "y" or train_or_not == "Y":

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training data to use for validation set
    valid_size = 0.2

    # writing a transform which have augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    # choose the training data and test set
    train_data = datasets.CIFAR10(r'C:\Users\arv78\Desktop\data',train=True,download=True,transform=transform)
    test_data = datasets.CIFAR10(r'C:\Users\arv78\Desktop\data',train=False,download=True,transform=transform)
        # data_dir = r'C:\Users\arv78\deep-learning-v2-pytorch\convolutional-neural-networks\cifar-cnn\data\cifar-10-batches-py'
        # train_data = datasets.ImageFolder(unpickle(data_dir) , transform=transform)
        # test_data = datasets.ImageFolder(unpickle(data_dir), transform=transform)

    # obtaining training indices to use for validation set
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx , valid_idx = indices[split:] , indices[:split]

    # define samplers for validation and training set
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,sampler=train_sampler,num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,sampler=valid_sampler,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)
    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    # helper function to un-normalize and display an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    # define the CNN architecture
    class Network(nn.Module):
        def __init__(self):
            super(Network,self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3,16,3,padding=1)
            self.conv1_1 = nn.Conv2d(16,16,3,padding=1)

            self.conv2 = nn.Conv2d(16,32,3,padding=1)
            self.conv2_1 = nn.Conv2d(32,32,3,padding=1)
            
            self.conv3 = nn.Conv2d(32,64,3,padding=1)
            self.conv3_1 = nn.Conv2d(64,64,3,padding=1)
            # self.conv3_2 = nn.Conv2d(64,64,3,padding=1)
            # self.conv3_3 = nn.Conv2d(256,256,3,padding=1)

            self.conv4 = nn.Conv2d(64,128,3,padding=1)
            self.conv4_1 = nn.Conv2d(128,128,3,padding=1)
            # self.conv4_2 = nn.Conv2d(128,128,3,padding=1)
            # self.conv4_3 = nn.Conv2d(512,512,3,padding=1)

            # max pooling layer
            self.pool = nn.MaxPool2d(2,2)
            # linear layers for fully connected layers
            self.fc1 = nn.Linear(128 * 2 * 2,500)
            self.fc2 = nn.Linear(500,10)
            # dropout layer
            self.dropout = nn.Dropout(0.25)

        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv1_1(x)))

            x = F.relu(self.conv2(x))
            x = self.pool(F.relu(self.conv2_1(x)))

            x = F.relu(self.conv3(x))
            # x = F.relu(self.conv3_1(x))
            x = self.pool(F.relu(self.conv3_1(x)))
            # x = self.pool(F.relu(self.conv3_3(x)))

            x = F.relu(self.conv4(x))
            # x = F.relu(self.conv4_1(x))
            x = self.pool(F.relu(self.conv4_1(x)))
            # x = self.pool(F.relu(self.conv4_3(x)))

            # flatten the image
            x = x.view(-1 , 128 * 2 * 2)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = Network()
    print(model)

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    # number of epochs to train the model
    num_epochs = 100
    # track change in validation loss
    valid_loss_min = np.Inf

    for epoch in range (1,num_epochs+1):
        
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        #training the model
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data , labels = data.cuda() , labels.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        # validate the model
        model.eval()
        for batch_idx, (data, labels) in enumerate(valid_loader):
            if train_on_gpu:
                data , labels = data.cuda() , labels.cuda()
            output = model(data)
            loss = criterion(output,labels)
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(),r'C:\Users\arv78\Desktop\model_augmented.pt')
            valid_loss_min = valid_loss

    # loading the model
    model.load_state_dict(torch.load(r'C:\Users\arv78\Desktop\model_augmented.pt'))
    # test the model
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    test_loss = 0.0
    accuracy = 0.0
    
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, labels) in enumerate(test_loader):
            if train_on_gpu:
                data , labels = data.cuda() , labels.cuda()
            output = model(data)
            loss = criterion(output,labels)
            test_loss += loss.item()*data.size(0)
            # calculate accuracy
                # top_p , top_class = output.topk(1,dim=1)
            _, top_class = torch.max(output, 1) 
                # equals = top_class == labels.view(*top_class.shape)
                # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            correct_tensor = top_class.eq(labels.data.view_as(top_class))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    ###################################
    # obtain one batch of test images #
    ##################################
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images.numpy()

    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        images = images.cuda()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(images.cpu()[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))

elif train_or_not == "n" or train_or_not == "N":
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20

    # writing a transform which have augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    # choose the training data and test set
    test_data = datasets.CIFAR10(r'C:\Users\arv78\Desktop\data',train=False,download=True,transform=transform)
        # data_dir = r'C:\Users\arv78\deep-learning-v2-pytorch\convolutional-neural-networks\cifar-cnn\data\cifar-10-batches-py'
        # test_data = datasets.ImageFolder(unpickle(data_dir), transform=transform)

    # prepare data loaders
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    # helper function to un-normalize and display an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    # define the CNN architecture
    class Network(nn.Module):
        def __init__(self):
            super(Network,self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3,16,3,padding=1)
            self.conv1_1 = nn.Conv2d(16,16,3,padding=1)

            self.conv2 = nn.Conv2d(16,32,3,padding=1)
            self.conv2_1 = nn.Conv2d(32,32,3,padding=1)
            
            self.conv3 = nn.Conv2d(32,64,3,padding=1)
            self.conv3_1 = nn.Conv2d(64,64,3,padding=1)
            # self.conv3_2 = nn.Conv2d(64,64,3,padding=1)
            # self.conv3_3 = nn.Conv2d(256,256,3,padding=1)

            self.conv4 = nn.Conv2d(64,128,3,padding=1)
            self.conv4_1 = nn.Conv2d(128,128,3,padding=1)
            # self.conv4_2 = nn.Conv2d(128,128,3,padding=1)
            # self.conv4_3 = nn.Conv2d(512,512,3,padding=1)

            # max pooling layer
            self.pool = nn.MaxPool2d(2,2)
            # linear layers for fully connected layers
            self.fc1 = nn.Linear(128 * 2 * 2,500)
            self.fc2 = nn.Linear(500,10)
            # dropout layer
            self.dropout = nn.Dropout(0.25)

        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv1_1(x)))

            x = F.relu(self.conv2(x))
            x = self.pool(F.relu(self.conv2_1(x)))

            x = F.relu(self.conv3(x))
            # x = F.relu(self.conv3_1(x))
            x = self.pool(F.relu(self.conv3_1(x)))
            # x = self.pool(F.relu(self.conv3_3(x)))

            x = F.relu(self.conv4(x))
            # x = F.relu(self.conv4_1(x))
            x = self.pool(F.relu(self.conv4_1(x)))
            # x = self.pool(F.relu(self.conv4_3(x)))

            # flatten the image
            x = x.view(-1 , 128 * 2 * 2)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = Network()
    print(model)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # loading the model
    model.load_state_dict(torch.load(r'C:\Users\arv78\Desktop\model_augmented.pt'))
    # test the model
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    test_loss = 0.0
    accuracy = 0.0
    
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, labels) in enumerate(test_loader):
            if train_on_gpu:
                data , labels = data.cuda() , labels.cuda()
            output = model(data)
            loss = criterion(output,labels)
            test_loss += loss.item()*data.size(0)
            # calculate accuracy
                # top_p , top_class = output.topk(1,dim=1)
            _, top_class = torch.max(output, 1) 
                # equals = top_class == labels.view(*top_class.shape)
                # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            correct_tensor = top_class.eq(labels.data.view_as(top_class))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    ###################################
    # obtain one batch of test images #
    ##################################
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images.numpy()

    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        images = images.cuda()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(images.cpu()[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))


