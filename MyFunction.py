import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456,0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456,0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456,0.406],
                                                               [0.229, 0.224, 0.225])]) 
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    trainloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloaders = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloaders, validloaders, testloaders

def train_model(model_select, hidden_units, lr, epochs, trainloaders, validloaders,device):
    from torchvision import models
    model = getattr(models, model_select)
    model = model(pretrained = True)
    # model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
# checked on pytorch vgg documentation, vgg classifier input feature is always 25088, so here we hard coded the input layer as 25088
# reference: https://pytorch.org/docs/0.4.0/_modules/torchvision/models/vgg.html
# Please help correct me if there is any misunderstanding. Thanks !
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval() # tell us the the layers
                with torch.no_grad():
                    for inputs,labels in validloaders:
                        inputs,labels = inputs.to(device), labels.to(device) 
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps,labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_category = ps.topk(1,dim=1)
                        equals = top_category == labels.view(*top_category.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Valid loss: {valid_loss/len(validloaders):.3f}.. "
                              f"Valid accuracy: {accuracy/len(validloaders):.3f}")
                    running_loss = 0
                    model.train() # tell us the model is trained
    return model, optimizer



def test_model(model, testloaders, device):
    criterion = nn.NLLLoss()
    test_loss = 0
    accuracy = 0
    model.eval() # tell us the the layers
    with torch.no_grad(): # close the gradient calculation
        for inputs,labels in testloaders:
            inputs,labels = inputs.to(device), labels.to(device) 
            logps = model.forward(inputs)
            batch_loss = criterion(logps,labels)
            test_loss += batch_loss.item()
            # calculate accuracy
            ps = torch.exp(logps)
            top_p, top_category = ps.topk(1,dim=1)
            equals = top_category == labels.view(*top_category.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(print(f"Test loss: {test_loss/len(testloaders):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloaders):.3f}"))
            running_loss = 0
            model.train() # tell us the model is trained


def save_model(save_directory, model, optimizer):
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, save_directory) 


def load_checkpoint(filepath, model_select):
    checkpoint = torch.load(filepath)
    from torchvision import models
    model = getattr(models, model_select)
    model = model(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # resize the image where the shortest side is 256 pixels
    
    width, height = image.size
    shortest_side = 256
    if width > height:
        ratio = 256/height
        new_size = int(width*ratio), 256
    else:
        ratio = 256/width
        new_size = 256, int(height*ratio)    
    image = image.resize(new_size)
    
    # crop out the center 224*224 portion of the image
    image = image.crop((width/2 - 112, height/2 - 112, width/2 + 112, height/2 + 112))
    
    # converst the value into array
    np_image = np.array(image)
    
    # convert the value from int 0-255 to float 0-1
    max_image = np.array(np_image.max(axis=(0,1)))
    min_image = np.array(np_image.min(axis=(0,1)))
    np_image = (np_image - min_image)/(max_image - min_image)
    
    # norrmalized the array
    mean =  np.array([0.485, 0.456, 0.406])
    std =  np.array([0.229,0.224,0.225])
    np_image = (np_image - mean)/std
    
    # adjust color channel to be the first channel
    np_image = np_image.transpose((2,0,1))
    return np_image
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    im = Image.open(image_path)
    np_image = process_image(im)
    inputs = torch.from_numpy(np_image)
    inputs.unsqueeze_(0)
    inputs = inputs.type(torch.FloatTensor)
    inputs = inputs.to(device)
    model.to(device)
    with torch.no_grad():
        logps = model.forward(inputs)
    ps = torch.exp(logps)
    probs, classes = ps.topk(5,dim=1)
    probs_result = probs.cpu()
    probs_result = probs_result.numpy()
    probs_result = probs_result[0]
    
    classes_result = classes.cpu()
    classes_result = classes_result.numpy()
    classes_result = classes_result[0]
    classes_result = list(map(str,classes_result))
    
    return probs_result, classes_result


