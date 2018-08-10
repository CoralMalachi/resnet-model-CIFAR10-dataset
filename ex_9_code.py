# coding=utf-8

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
# import autograd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

#define:
INPUT = 3 #
NUM_OF_FILTERS = 6
FILTER_SIZE = 5
LEARNING_RATE = 0.01
SIZE_BATCH = 64
NUM_OF_EPOCHS = 10



###############################################################
# convolution layer : we have a number of filters or kernels. These filters are just small
# patches that represent some kind of visual feature these are the weights and
# biases of a CNN. We take each filter and convolve it over the input volume to
#get a single activation map

#pooling layer : This layer is primarily used to help reduce the computational complexity and
#extract prominent features. we have to define a pool size, which tells us by how much we
#should reduce the width and height of the activation volume  The most common pooling size is 2 * 2
#reduce the spatial size of the representation to reduce the amount of parameters and computation in the network
#The most common approach used in pooling is max pooling.

#Fully-Connected Layer -This layer is just a plain-old artificial neural network! The catch is that the input
#must be a vector. So if we have an activation volume, we must flatten it into a
#vector first! After flattening the volume, we can treat this layer just like a neural network


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #initial size 32 x 32 x 1
        #convolution layer - For the first layer, we have 6 filters of size 5×5
        #size after first conv 28x28x6

        self.first_conv = nn.Conv2d(INPUT,NUM_OF_FILTERS,FILTER_SIZE)

        self.relu1 = nn.ReLU()

        #pooling layer - we perform max pooling using 2×2 regions
        #size after pool 14x14x6
        self.pool1 = nn.MaxPool2d(2,2)

        # convolution layer - we repeat that combination except using 16 filters for the next block
        #size after first conv 10x10x16
        self.second_conv = nn.Conv2d(6,16,5)

        self.relu2 = nn.ReLU()

        # pooling layer - we perform max pooling using 2×2 regions
        # size after pool 5x5x16
        self.pool2 = nn.MaxPool2d(2,2)

        # Fully-Connected Layer
        self.fc1 = nn.Linear(16*5*5,120)

        self.relu3 = nn.ReLU()

        self.fc2 =nn.Linear(120,84)

        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(84,10)

        self.batch = nn.BatchNorm2d(6)

        self.batch1 = nn.BatchNorm1d(120)

        self.batch2 = nn.BatchNorm1d(84)


    def forward(self, x):
        x=self.first_conv(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.batch(x)

        x = self.second_conv(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1,400)

        x = self.fc1(x)

        x = self.batch1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = self.batch2(x)

        x = F.relu(x)

        x = self.fc3(x)

        return F.log_softmax(x,dim=1)





###############################################################
#Function Name:print_message_each_epoch
#Function input:kind_of_set,length of set, loss of model, number
#of correct predictions of model and size of batch
#Function output:none
#Function Action:the function print a message to help the user
#follow the network progress
################################################################

def print_message_each_epoch(kind_of_set,set_len,m_loss,m_success,size_of_batch):
    print('\n' + kind_of_set + ': The average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        m_loss, m_success, (set_len * size_of_batch),
        100. * m_success / (set_len * size_of_batch)))

###############################################################
#Function Name: calculate_loss_print
#Function input: size_of_batch,model, set,is_training-boolean varible
#indicates id the set is trainning set or validation set
#Function output:loss
#Function Action:the function calculate the loss of the model
#for each epochs,and print write message
################################################################

def calculate_loss_print(size_of_batch,model,set,is_training):

    #boolean varible indicates id the set is trainning set or validation set
    print_kind_of_set="training set"
    if is_training != 1:
        print_kind_of_set = "validation set"
    #define varibles for loss, and number of correct prediction
    m_loss=0
    m_success=0
    #let the model know to switch to eval mode by calling .eval() on the model
    model.eval()
    for data,tag in set:
        #feed model with data
        model_result = model(data)
        #sum the loss
        m_loss = m_loss + F.nll_loss(model_result,tag,size_average=False).item()
        #call help function to get the right prediction
        y_tag = get_y_tag(model_result)
        #total of successfull predictions
        m_success += y_tag.eq(tag.data.view_as(y_tag)).cpu().sum()
    # save the len of training set in varible to save calls to len functions
    set_len = len(set)
    #calculate loss
    m_loss = m_loss/(size_of_batch*set_len)
    #call help function to print message about loss each epoch
    print_message_each_epoch(print_kind_of_set,set_len,m_loss,m_success,size_of_batch)
    return m_loss


###############################################################
#Function Name: get_y_tag
#Function input: model
#Function output:return the model prediction tag
#Function Action: the function return the prediction
#by getting  the index of the max log-probability
################################################################

def get_y_tag(model):
    return model.data.max(1, keepdim=True)[1]


def create_predictions_file(model, set):

    predictions_list = []
    correct_tags_list = []
    results_file = open("test.pred", 'w')
    # let the model know to switch to eval mode by calling .eval() on the model
    model.eval()
    m_loss = 0
    #count number of success predictions
    num_of_success = 0
    for data, target in set:
        output = model(data)
        #sum the loss
        m_loss = m_loss + F.nll_loss(output, target, size_average=False).item()
        #call get_y_tag function to get prediction
        y_tag = get_y_tag(output)
        #update varible
        num_of_success = num_of_success + y_tag.eq(target.data.view_as(y_tag)).cpu().sum()

        correct_tags_list.append(target.item())

        predictions_list.append(y_tag.item())
        #write to file current prediction according to the required format
        results_file.write(str(y_tag.item()) + "\n")
    #save the len of training set in varible to save calls to len functions
    set_len = len(set)
    #calaculate the final loss
    m_loss = m_loss / (set_len)

    print('\n Test_Set: the Average loss: {:.4f}, the Accuracy: {}/{} ({:.0f}%)\n'.format(
        m_loss, num_of_success, (set_len), 100. * num_of_success / (set_len)))
    #create the confusion matrix
    conf_matrix=confusion_matrix(correct_tags_list,predictions_list)
    print(conf_matrix)
    #close prediction file
    results_file.close()




###############################################################
#Function Name:train_neural_network
#Function input:model, trainng and validation sets
#Function output:none
#Function Action:train on the training set and then test the
#network on the test set. This has the network make predictions on data it has never seen
################################################################

def train_neural_network(model,train,validation_set):

    #define 2 empty list to sve there the loss
    validation_set_scores = {}
    train_set_scores={}
    #set the optimizer
    optimizer=optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_OF_EPOCHS):
        print "epoch number "+ str(epoch)
        model.train()

        for data, labels in train:
            optimizer.zero_grad()
            output = model(data)
            running_los = F.nll_loss(output,labels)
            running_los.backward()
            optimizer.step()
        #calculate the loss by calling calculate_loss_print function
        running_los = calculate_loss_print(SIZE_BATCH,model,train,1)
        train_set_scores[epoch+1]=running_los
        # calculate the loss by calling calculate_loss_print function
        running_los = calculate_loss_print(1,model,validation_set,0)
        validation_set_scores[epoch+1]=running_los
    first_lable, = plt.plot(validation_set_scores.keys(), validation_set_scores.values(), "g-", label='validation loss')
    second_lable, = plt.plot(train_set_scores.keys(), train_set_scores.values(), "r-", label='train loss')
    plt.legend(handler_map={first_lable: HandlerLine2D(numpoints=4)})
    plt.show()



###############################################################
#Function Name:load_datasets_and_run_models
#Function input:none
#Function output:none
#Function Action:the function use torchvision inorder to uplode
#the MNIST training and test datasets. then,split the data (80:20)
#and build the network.
################################################################
def load_datasets_and_run_models():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    # Load the CIFAR10 training and test datasets using torchvision
    training_set = datasets.CIFAR10(root='./data',train=True, download=True,transform=transform)
    #save the len of training set in varible to save calls to len functions
    training_set_len = len(training_set)
    divided_size = int(0.2* training_set_len)

    set_test = datasets.CIFAR10(train=False,root='./data',transform=transform)

    train_list = list(range(training_set_len))

    #random, non-contigous split
    validation_idx = np.random.choice(train_list,size=divided_size,replace=False)
    training_idx= list(set(train_list)-set(validation_idx))

    samples_train = SubsetRandomSampler(training_idx)
    samples_validation = SubsetRandomSampler(validation_idx)

    m_training_loader = torch.utils.data.DataLoader(batch_size=SIZE_BATCH,sampler=samples_train,dataset=training_set)
    m_validation_loader = torch.utils.data.DataLoader(batch_size=1,sampler=samples_validation,dataset=training_set)

    m_test_loader = torch.utils.data.DataLoader(dataset=set_test,shuffle=False,batch_size=1)
    #creat netC
    model = CNN()
    #call the main function of the program - training our network
    train_neural_network(model,m_training_loader,m_validation_loader)
    #call help function which create a predictions file
    create_predictions_file(model,m_test_loader)


if __name__ == "__main__":
    load_datasets_and_run_models()


############################# RESNET MODEL : #############################################


#
#
# def create_predictions_file(model, set,criterion):
#
#     predictions_list = []
#     correct_tags_list = []
#     results_file = open("test.pred", 'w')
#     # let the model know to switch to eval mode by calling .eval() on the model
#     model.eval()
#     m_loss = 0
#     #count number of success predictions
#     num_of_success = 0
#     for data, target in set:
#         output = model(data)
#         #sum the loss
#         m_loss = m_loss + F.nll_loss(output, target, size_average=False).item()
#         #call get_y_tag function to get prediction
#         y_tag = get_y_tag(output)
#         #update varible
#         num_of_success = num_of_success + y_tag.eq(target.data.view_as(y_tag)).cpu().sum()
#
#         correct_tags_list.append(target.item())
#
#         predictions_list.append(y_tag.item())
#         #write to file current prediction according to the required format
#         results_file.write(str(y_tag.item()) + "\n")
#     #save the len of training set in varible to save calls to len functions
#     set_len = len(set)
#     #calaculate the final loss
#     m_loss = m_loss / (set_len)
#
#     print('\n Test_Set: the Average loss: {:.4f}, the Accuracy: {}/{} ({:.0f}%)\n'.format(
#         m_loss, num_of_success, (set_len), 100. * num_of_success / (set_len)))
#
#     conf_matrix=confusion_matrix(correct_tags_list,predictions_list)
#     print(conf_matrix)
#     #close prediction file
#     results_file.close()
#
# def train_neural_network(optimizer,criterion,model,train,validation_set):
#
#     #define 2 empty list to sve there the loss
#     validation_set_scores = {}
#     train_set_scores={}
#     #set the optimizer
#     optimizer=optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
#     for epoch in range(NUM_OF_EPOCHS):
#         print "epoch number "+ str(epoch)
#         model.train()
#
#         for data, labels in train:
#             optimizer.zero_grad()
#             output = model(data)
#             running_los = criterion(output,labels)
#             running_los.backward()
#             optimizer.step()
#         #calculate the loss by calling calculate_loss_print function
#         running_los = calculate_loss_print(SIZE_BATCH,model,train,1,criterion)
#         train_set_scores[epoch+1]=running_los
#         # calculate the loss by calling calculate_loss_print function
#         running_los = calculate_loss_print(1,model,validation_set,0,criterion)
#         validation_set_scores[epoch+1]=running_los
#     first_lable, = plt.plot(validation_set_scores.keys(), validation_set_scores.values(), "g-", label='validation loss')
#     second_lable, = plt.plot(train_set_scores.keys(), train_set_scores.values(), "r-", label='train loss')
#     plt.legend(handler_map={first_lable: HandlerLine2D(numpoints=4)})
#     plt.show()
#
#
#
# ###############################################################
# #Function Name:load_datasets_and_run_models
# #Function input:none
# #Function output:none
# #Function Action:the function use torchvision inorder to uplode
# #the MNIST training and test datasets. then,split the data (80:20)
# #and build the network.
# ################################################################
# def load_datasets_and_run_models():
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))])
#     # Load the CIFAR10 training and test datasets using torchvision
#     training_set = datasets.CIFAR10(root='./data'
#                                     ,train=True,
#                                     download=True,
#                                     transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
#     #save the len of training set in varible to save calls to len functions
#     training_set_len = len(training_set)
#     divided_size = int(0.2* training_set_len)
#
#     set_test = datasets.CIFAR10(train=False,root='./data',
#                                 transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
#
#     train_list = list(range(training_set_len))
#
#     #random, non-contigous split
#     validation_idx = np.random.choice(train_list,size=divided_size,replace=False)
#     training_idx= list(set(train_list)-set(validation_idx))
#
#     samples_train = SubsetRandomSampler(training_idx)
#     samples_validation = SubsetRandomSampler(validation_idx)
#
#     m_training_loader = torch.utils.data.DataLoader(batch_size=SIZE_BATCH,sampler=samples_train,dataset=training_set)
#     m_validation_loader = torch.utils.data.DataLoader(batch_size=1,sampler=samples_validation,dataset=training_set)
#
#     m_test_loader = torch.utils.data.DataLoader(dataset=set_test,shuffle=False,batch_size=1)
#
#     model = models.resnet18(pretrained=True)
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Parameters of newly constructed modules have requires_grad=True by default
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 10)
#
#
#     optimizer = optim.Adagrad(model.fc.parameters(), lr=LEARNING_RATE)
#
#     m_criterion = nn.CrossEntropyLoss()
#
#     train_neural_network(optimizer,m_criterion,model,m_training_loader,m_validation_loader)
#
#
#     create_predictions_file(model, m_test_loader, m_criterion)

