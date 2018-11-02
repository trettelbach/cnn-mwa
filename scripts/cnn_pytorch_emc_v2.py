import torch
import torchvision.transforms as transforms
import torch.nn as nn
import astropy.io.fits as pyfits
import torch.utils.data as data
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support
from time import gmtime, strftime
#import mylogger
import mytransforms
import myplotting
from StringIO import StringIO

#try:
#    from StringIO import StringIO  # Python 2.7
#except ImportError:
#    from io import BytesIO  # Python 3.x

NLABEL = int(5)
BATCH_SIZE = 1
EPOCHS = 3
INIT_LR = 5e-7

use_gpu = torch.cuda.is_available()
# use_gpu = False
print(use_gpu)

device = torch.device("cuda:0" if use_gpu else "cpu")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))


transform = transforms.Compose([
    mytransforms.CenterCropFits(800),
    # CutRangeFits(0),
    mytransforms.RandomHorizontalFlipFits(),
    mytransforms.RandomVerticalFlipFits(),
    mytransforms.ToTensorFits()
    ])

labels = []
train_loss_list = []
test_loss_list = []

class MyCustomDatasetFits(data.Dataset):
    # __init__ function is where the initial logic happens like reading a csv,
    # assigning transforms etc.
    def __init__(self, csv_path):
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # the rest contain the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1:])
        labels.append(self.label_arr)
        # Calculate len
        self.data_len = len(self.data_info.index)

    # __getitem__ function returns the data and labels. This function is
    # called from dataloader like this
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        data1 = pyfits.open(single_image_name, axes=2)
        data2 = data1[0].data.astype('float32')
        data3 = data2.reshape(4000, 4000, 1)

        img_mean = np.ndarray.mean(data3)
        img_std = np.std(data3)
        img_stand = (data3 - img_mean)/img_std
        img_as_tensor = transform(img_stand)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label, single_image_name

    def __len__(self):
        return self.data_len


# load training set, data augmentation
fits_train = MyCustomDatasetFits('/media/tabea/FIRENZE/cnn-mwa/data/labels_train.csv')
trainloader = torch.utils.data.DataLoader(fits_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

# load testing set, data augmentation
fits_test = MyCustomDatasetFits('/media/tabea/FIRENZE/cnn-mwa/data/labels_test.csv')
testloader = torch.utils.data.DataLoader(fits_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

pictures = []
class CNN(nn.Module):
    def __init__(self, nlabel):
        super(CNN, self).__init__()
        self.nlabel = nlabel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # print(self.layer1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.ReLU()
        self.fc1 = nn.Linear(80000, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, NLABEL)
        # self.apply(weights_init)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.DataParallel(self.fc1)
        self.fc2 = nn.DataParallel(self.fc2)
        self.fc3 = nn.DataParallel(self.fc3)

    def forward(self, x):
        out = self.layer1(x)
        out1 = out[0][0].cpu().detach().numpy()
        out = self.layer2(out)
        out2 = out[0][0].cpu().detach().numpy()
        out = self.layer3(out)
        out3 = out[0][0].cpu().detach().numpy()
        out = self.layer4(out)
        out4 = out[0][0].cpu().detach().numpy()
        out = self.layer5(out)
        # print(out.weight)
        out = out.view(out.size(0), -1)  # flattening... out.view(BATCH_SIZE, -1)
        
        out = self.fc1(out)  # fully connected layers at the end
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.dropout(out)
        pictures.append(out4)
        # show_images(pictures, 2, titles=['1', '2', '3', '4'])
        # plt.show()
        return out, pictures

# instance of the Conv Net

# model = CNN(NLABEL)
model = CNN(NLABEL).to(device)
#logger = mylogger.Logger('./logs')

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss(reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

train_losses = []
test_losses = []

train_accs = []
test_accs = []


train_true = []
train_pred = []

test_true = []
test_pred = []


for epoch in range(EPOCHS):
    model.train()
    correct_train = 0
    for i, (images, labels, location) in enumerate(trainloader):
        if use_gpu:
            images = Variable(images.float().cuda(), volatile=False)
            labels = Variable(labels.cuda(), volatile=False)

        else:
            images = Variable(images.float(), requires_grad=True, volatile=False)
            labels = Variable(labels, volatile=False)

        optimizer.zero_grad()
        outputs = model(images)[0]
        m = nn.Softmax()
        probs = m(outputs)
        max_values, max_indices = torch.max(probs[0], 0)

        print('correct: %d - predicted: %d at %.3f %% - image: %s' % (labels.item(), max_indices.item(), max_values.item()*100, location))

        if labels.item() == max_indices.item():
            correct_train += 1
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()

        train_losses.append(float(str(loss.data[0])[7:-18]))

        if (i + 1) % 100 == 0:
            print('TRAINING: Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, EPOCHS, i + 1, len(fits_train) // BATCH_SIZE, loss.data[0]))

        if epoch == (EPOCHS - 1):
            train_true.append(labels.item())
            train_pred.append(max_indices.item())


    print('Accuracy epoch %d: %.2f' % (epoch+1, 100 * correct_train / 1047))
    train_accs.append(100*correct_train/1047)
#    print(losses)

    # set network to evaluation mode (no more dropout, fi)
    model.eval()
    correct = 0
    total = 0

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)
    #info = {'loss': float(str(loss.data[0])[7:-18]), 'accuracy': (100*correct_train/12)}

    #for tag, value in info.items():
    #    logger.scalar_summary(tag, value, epoch + 1)

    # 2. Log values and gradients of the parameters (histogram summary)
    #for tag, value in model.named_parameters():
    #    tag = tag.replace('.', '/')
    #    logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
    #    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

    # 3. Log training images (image summary)
    #info = {'images': images.view(-1, 800, 800)[:10].cpu().numpy()}

    #for tag, images in info.items():
    #    logger.image_summary(tag, images, epoch + 1)

    # ================================================================== #
    #                      Tensorboard Logging Over                      #
    # ================================================================== #

    for images, labels, location in testloader:
        if use_gpu:
            images = Variable(images.float().cuda(), requires_grad=True, volatile=False)
            labels = Variable(labels.cuda(), volatile=False)

        else:
            images = Variable(images.float(), requires_grad=True, volatile=False)
            labels = Variable(labels, volatile=False)

        outputs = model(images)[0]
        m = nn.Softmax()
        probs = m(outputs)
        max_values, max_indices = torch.max(probs[0], 0)
        test_loss = criterion(outputs, labels.squeeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        test_losses.append(float(str(test_loss.data[0])[7:-18]))
        print('correct: %d - predicted: %d at %.3f %% - image: %s' % (labels.item(), max_indices.item(), max_values.item() * 100, location))
        print('Loss: %.4f - image: %s' % (test_loss.data[0], location))
        if epoch == (EPOCHS - 1):
            test_true.append(labels.item())
            test_pred.append(max_indices.item())

    print('Test Accuracy of the model on the %d test images after %d: %.2f %%' % (total, epoch+1, 100 * correct / (total*BATCH_SIZE)))
    test_accs.append(float(str(100*correct/(total*BATCH_SIZE))[7:-18]))

print(train_losses)
print(test_losses)
print(train_accs)
print(test_accs)
print('train_true: ', train_true)
print('train_pred: ', train_pred)
print('test_true: ', test_true)
print('test_pred: ', test_pred)

show_images(pictures[24:], 3)  #, titles=['1', '2', '3', '4'])

# https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/cnn_layer_visualization.py
# das koennte funktionieren... mal Model speichern und ausprobieren

#print(precision_recall_fscore_support(test_true, test_pred))
#cm = confusion_matrix(test_true, test_pred)
#class_names = ['0', '1', '2', '3', '4']
#plt.figure()
#plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Confusion matrix, with normalization')
#plt.show()

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
torch.save(model, '/media/tabea/FIRENZE/cnn-mwa/results/model001')

