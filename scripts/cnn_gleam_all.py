import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from cnn_gleam import CnnGleam
import astropy.io.fits as pyfits
import myreader
import time
import sklearn.metrics
import sys
import argparse


def main():
    """ Trains a neural model with the CnnGleam architecture,
    saves the model weights,
    and prints some statistics, if desired
    """

    parser = argparse.ArgumentParser(description='Train and test the GleamNet '
                                                 'for Classification')
    parser.add_argument('path', metavar='PATH', type=str, nargs='?',
                        help='path to the csv with file locations')
    parser.add_argument('statistics', metavar='STATS', type=int, nargs='?',
                        help='determine if statistics should be calculated')
    args = parser.parse_args()
    print(args.statistics)

    # set path to data, number of dataset labels,
    # number of epochs to train, initial learning rate
    PATH = args.path
    NLABEL = 3
    EPOCHS = 5
    INIT_LR = 5e-7
    VER = 1.0

    # set time
    start_time = time.time()

    # check if GPU available, and set device accordingly
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # do I want statistics?
    if args.statistics == 0:
        stats = False
        sys.stdout.write('training the model only\n')
    else:
        stats = True
        sys.stdout.write('training the model and producing further '
                         'statistics\n')

    # load training set, data augmentation
    fits_train = myreader.MyCustomDatasetFits(
        PATH + 'mwa_data/archiv/pytorch_container/labels_train.csv',
        transformation='train')
    trainloader = torch.utils.data.DataLoader(fits_train,
                                              shuffle=True,
                                              num_workers=1)
    # load testing set, data augmentation
    fits_test = myreader.MyCustomDatasetFits(
        PATH + 'mwa_data/archiv/pytorch_container/labels_train.csv',
        transformation='test')
    testloader = torch.utils.data.DataLoader(fits_test,
                                             shuffle=True,
                                             num_workers=1)

    # make instance of the Conv Net, send to GPU if available
    model = CnnGleam(NLABEL)
    if use_gpu:
        model = model.cuda()

    # initialize the architecture for the model
    # model = cnn_gleam.CnnGleam(NLABEL).to(device)

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # prepare some lists for statistical evaluation
    if stats:
        train_losses = []
        train_true = []
        train_pred = []
        train_cert = []
        train_f1 = []
        train_loc = []
        train_acc = []

        test_losses = []
        test_true = []
        test_pred = []
        test_cert = []
        test_f1 = []
        test_loc = []
        test_acc = []

    # run the data through the network for x epochs
    for epoch in range(EPOCHS):
        # set model to training mode
        model.train()
        correct_train = 0

        # iterate through the trainig set, send tensors to GPU/CPU
        for i, (images, labels, location) in enumerate(trainloader):
            if use_gpu:
                labels = Variable(labels.cuda(), volatile=False)
                images = Variable(images.float().cuda(),
                                  requires_grad=False,
                                  volatile=False)
            else:
                labels = Variable(labels, volatile=False)
                images = Variable(images.float(),
                                  requires_grad=False,
                                  volatile=False)

            # clear gradients of all optimized tensors
            optimizer.zero_grad()

            # run the model and apply softmax at the end
            outputs = model(images)
            m = nn.Softmax(dim=1)
            probs = m(outputs)

            # get predictions
            max_values, max_indices = torch.max(probs[0], 0)

            sys.stdout.write('correct: %d - predicted: %d at %.3f %% - '
                             'image: %s\n'
                             % (labels.item(), max_indices.item(),
                                max_values.item() * 100, location))

            # calculate loss
            train_loss = criterion(outputs, labels.squeeze(1))

            # parameter update based on the current gradient
            train_loss.backward()
            optimizer.step()

            if stats:
                correct_train += (max_indices == labels).sum().item()
                train_losses.append(train_loss.item())
                train_true.append(labels.item())
                train_pred.append(max_indices.item())
                train_loc.append(location)
                train_cert.append(max_values.item())

        if stats:
            train_true_last = train_true[-len(trainloader):]
            train_pred_last = train_pred[-len(trainloader):]
            print(train_true)
            print(train_pred)
            print(train_true_last)
            print(train_pred_last)
            train_acc.append(correct_train / float(len(trainloader)))
            train_f1.append(sklearn.metrics.f1_score(train_true_last,
                                                     train_pred_last,
                                                     average='micro'))

        # set model to evaluation mode
        model.eval()
        correct_test = 0

        # iterate through the testing set, send tensors to GPU/CPU
        for images, labels, location in testloader:
            if use_gpu:
                labels = Variable(labels.cuda(), volatile=False)
                images = Variable(images.float().cuda(),
                                  requires_grad=True,
                                  volatile=False)

            else:
                labels = Variable(labels, volatile=False)
                images = Variable(images.float(),
                                  requires_grad=True,
                                  volatile=False)

            # run the model and apply SoftMax at the end
            outputs = model(images)
            m = nn.Softmax(dim=1)
            probs = m(outputs)
            # get predictions
            max_values, max_indices = torch.max(probs[0], 0)

            # calculate loss
            if stats:
                test_loss = criterion(outputs, labels.squeeze(1))
                test_losses.append(test_loss.item())
                correct_test += (max_indices == labels).sum().item()

            # write the results to the fits-header
            hdulist = pyfits.open(location[0])
            img_header = hdulist[0].header
            img_header.set('cnnver', VER)
            img_header.set('good', probs[0][0].item())
            img_header.set('rfi', probs[0][1].item())
            img_header.set('sis', probs[0][2].item())
            img_header.set('rfisis', 'NA')  # TODO: anpassen
            hdulist.writeto(location[0], overwrite=True)
            hdulist.close()

            sys.stdout.write(
                'correct: %d - predicted: %d at %.3f %% - image: %s\n'
                % (labels.item(), max_indices.item(),
                   max_values.item() * 100, location))

            if stats:
                test_losses.append(test_loss.item())
                test_loc.append(location)
                test_true.append(labels.item())
                test_pred.append(max_indices.item())
                test_cert.append(max_values.item())

        if stats:
            test_true_last = test_true[-len(testloader):]
            test_pred_last = test_pred[-len(testloader):]
            print(test_true)
            print(test_pred)
            print(test_true_last)
            print(test_pred_last)
            test_acc.append(correct_test / float(len(testloader)))
            test_f1.append(sklearn.metrics.f1_score(test_true_last,
                                                    test_pred_last,
                                                    average='micro'))

    # write out trainjng statistics into extra files (after all epochs)
    if stats:
        with open(PATH + 'train_losses.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_losses)
        with open(PATH + 'train_true.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_true)
        with open(PATH + 'train_pred.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_pred)
        with open(PATH + 'train_cert.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_cert)
        with open(PATH + 'train_f1.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_f1)
        with open(PATH + 'train_loc.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_loc)
        with open(PATH + 'train_acc.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_acc)

        with open(PATH + 'test_losses.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_losses)
        with open(PATH + 'test_true.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_true)
        with open(PATH + 'test_pred.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_pred)
        with open(PATH + 'test_cert.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_cert)
        with open(PATH + 'test_f1.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_f1)
        with open(PATH + 'test_loc.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_loc)
        with open(PATH + 'test_acc.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_acc)

    # save the model weights
    torch.save(model.state_dict(), PATH + 'gleamnet')

    sys.stdout.write('Duration of training and testing (in s): %.2f\n'
                     % (time.time() - start_time))


if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn
# import torch.utils.data as data
# import astropy.io.fits as pyfits
# from torch.autograd import Variable
# import time
# from cnn_gleam import CnnGleam
# import myreader
# # import sklearn.metrics
# import sys
# import numpy as np
#
#
# def main():
#     """ Trains a neural model with the CnnGleam architecture,
#     saves the model weights,
#     and prints some statistics, if desired
#     """
#
#     # set path to data, number of dataset labels,
#     # number of epochs to train, initial learning rate
#     PATH = '/media/tabea/FIRENZE/'
#     NLABEL = int(3)
#     EPOCHS = 3
#     INIT_LR = 5e-7
#     VER = 1.0
#
#     # check if GPU available
#     use_gpu = torch.cuda.is_available()
#
#     start_time = time.time()
#
#     # do I want statistics?
#     # TODO: find out, if you should specify this in the run statement
#     # TODO: (via stdin, oder so)
#     # stats = True
#
#     # load training set, data augmentation
#     fits_train = myreader.MyCustomDatasetFits(
#         PATH + 'mwa_data/archiv/pytorch_container/labels_train.csv',
#         transformation='train')
#     trainloader = torch.utils.data.DataLoader(fits_train,
#                                               shuffle=True,
#                                               num_workers=1)
#     # load training set, data augmentation
#     fits_test = myreader.MyCustomDatasetFits(
#         PATH + 'mwa_data/archiv/pytorch_container/labels_test02.csv',
#         transformation='test')
#     testloader = torch.utils.data.DataLoader(fits_test,
#                                              shuffle=True,
#                                              num_workers=1)
#
#     # make instance of the Conv Net, send to GPU if available
#     model = CnnGleam(NLABEL)
#     if use_gpu:
#         model = model.cuda()
#
#     # define criterion and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
#
#     # prepare some lists for statistical evaluation
#     zerozero_avg = []
#     zeroone_avg = []
#     zerotwo_avg = []
#     onezero_avg = []
#     oneone_avg = []
#     onetwo_avg = []
#     twozero_avg = []
#     twoone_avg = []
#     twotwo_avg = []
#
#     # TODO: ab hier approved
#     test_losses = []
#     train_losses = []
#     train_f1 = []
#     test_f1 = []
#     train_cert = []
#     test_cert = []
#     train_loc = []
#     test_loc = []
#     train_accs = []
#     test_accs = []
#
#     # run the data through the network for x epochs
#     for epoch in range(EPOCHS):
#         # set model to training mode
#         model.train()
#         # TODO: ab hier approved
#         train_true = []
#         train_pred = []
#
#         test_true = []
#         test_pred = []
#
#         zerozero = []
#         zeroone = []
#         zerotwo = []
#         onezero = []
#         oneone = []
#         onetwo = []
#         twozero = []
#         twoone = []
#         twotwo = []
#
#         correct_train = 0
#         correct_test = 0
#
#         # iterate through the trainig set, send tensors to GPU/CPU
#         for i, (images, labels, location) in enumerate(trainloader):
#             if use_gpu:
#                 labels = Variable(labels.cuda(), volatile=False)
#                 images = Variable(images.float().cuda(),
#                                   requires_grad=False,
#                                   volatile=False)
#             else:
#                 labels = Variable(labels, volatile=False)
#                 images = Variable(images.float(),
#                                   requires_grad=False,
#                                   volatile=False)
#
#             # clear gradients of all optimized tensors
#             optimizer.zero_grad()
#
#             # run the model and apply softmax at the end
#             outputs = model(images)
#             m = nn.Softmax(dim=1)
#             probs = m(outputs)
#             x = probs.cpu().detach().numpy().squeeze()
#
#             a = x[0]
#             b = x[1]
#             c = x[2]
#
#             # das ist eigentlich nur fuer das wissenschaftliche
#             if labels.item() == 0:
#                 zerozero.append(a)
#                 zeroone.append(b)
#                 zerotwo.append(c)
#             elif labels.item() == 1:
#                 onezero.append(a)
#                 oneone.append(b)
#                 onetwo.append(c)
#             elif labels.item() == 2:
#                 twozero.append(a)
#                 twoone.append(b)
#                 twotwo.append(c)
#
#             # get predictions
#             max_values, max_indices = torch.max(probs[0], 0)
#
#             # sys.stdout.write('correct: %d - predicted: %d at %.3f %% - image: %s\n'
#             #                  % (labels.item(), max_indices.item(),
#             #                     max_values.item() * 100, location))
#
#             # calculate loss
#             train_loss = criterion(outputs, labels.squeeze(1))
#
#             # parameter update based on the current gradient
#             train_loss.backward()
#             optimizer.step()
#
#             # create some contents for statistical evaluation
#             train_losses.append(train_loss.item())
#             train_true.append(labels.item())
#             train_pred.append(max_indices.item())
#             if epoch == (EPOCHS - 1):
#                 train_loc.append(location)
#             train_cert.append(max_values.item())
#             if labels.item() == max_indices.item():
#                 correct_train += 1
#
#         train_accs.append(100 * correct_train // len(train_true))
#         # train_f1.append(sklearn.metrics.f1_score(train_true,
#         #                                              train_pred,
#         #                                              average='micro'))
#
#         zerozero_avg.append(np.mean(zerozero))
#         zeroone_avg.append(np.mean(zeroone))
#         zerotwo_avg.append(np.mean(zerotwo))
#         onezero_avg.append(np.mean(onezero))
#         oneone_avg.append(np.mean(oneone))
#         onetwo_avg.append(np.mean(onetwo))
#         twozero_avg.append(np.mean(twozero))
#         twoone_avg.append(np.mean(twoone))
#         twotwo_avg.append(np.mean(twotwo))
#
#         # set model to evaluation mode
#         model.eval()
#
#         # iterate through the testing set, send tensors to GPU/CPU
#         for images, labels, location in testloader:
#             if use_gpu:
#                 labels = Variable(labels.cuda(), volatile=False)
#                 images = Variable(images.float().cuda(),
#                                   requires_grad=True,
#                                   volatile=False)
#
#             else:
#                 labels = Variable(labels, volatile=False)
#                 images = Variable(images.float(),
#                                   requires_grad=True,
#                                   volatile=False)
#
#             # run the model and apply SoftMax at the end
#             outputs = model(images)
#             m = nn.Softmax(dim=1)
#             probs = m(outputs)
#             # get predictions
#             max_values, max_indices = torch.max(probs[0], 0)
#
#             # calculate loss
#             test_loss = criterion(outputs, labels.squeeze(1))
#             # print(labels.size(0))
#             # correct += (max_indices == labels).sum()
#             test_losses.append(test_loss.item())
#             # print(location[0])
#
#             # write the results to the fits-header
#             hdulist = pyfits.open(location[0])
#             img_header = hdulist[0].header
#             img_header.set('cnnver', VER)
#             img_header.set('good', probs[0][0].item())
#             img_header.set('rfi', probs[0][1].item())
#             img_header.set('sis', probs[0][2].item())
#             img_header.set('rfisis', 'NA')  # TODO: anpassen
#             hdulist.writeto(location[0], overwrite=True)
#             hdulist.close()
#
#             sys.stdout.write(
#                 'correct: %d - predicted: %d at %.3f %% - image: %s\n'
#                 % (labels.item(), max_indices.item(),
#                    max_values.item() * 100, location))
#
#
#             # # test_losses.append(test_loss.item())
#             test_true.append(labels.item())
#             test_pred.append(max_indices.item())
#             if epoch == (EPOCHS - 1):
#                 test_loc.append(location)
#             test_cert.append(max_values.item())
#             if labels.item() == max_indices.item():
#                 correct_test += 1
#
#         test_accs.append(100 * correct_test // len(test_true))
#         # test_f1.append(sklearn.metrics.f1_score(test_true,
#         #                                         test_pred,
#         #                                         average='micro'))
#
#     # write out statistics into extra files
#     with open(PATH + 'train_losses.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_losses)
#     with open(PATH + 'train_true.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_true)
#     with open(PATH + 'train_pred.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_pred)
#     with open(PATH + 'train_cert.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_cert)
#     with open(PATH + 'train_f1.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_f1)
#     with open(PATH + 'train_loc.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_loc)
#     with open(PATH + 'train_acc.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in train_accs)
#
#     with open('/media/tabea/FIRENZE/test_losses.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_losses)
#     with open('/media/tabea/FIRENZE/test_true.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_true)
#     with open('/media/tabea/FIRENZE/test_pred.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_pred)
#     with open('/media/tabea/FIRENZE/test_cert.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_cert)
#     with open('/media/tabea/FIRENZE/test_f1.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_f1)
#     with open(PATH + 'test_loc.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_loc)
#     with open('/media/tabea/FIRENZE/test_accs.txt', 'w') as ff:
#         ff.writelines("%s\n" % elem for elem in test_accs)
#
#     print(test_accs)  # passt, zu datei
#     print(len(test_accs))
#
#     print('Duration of training + testing (in s): ' + str(time.time() -
#                                                       start_time))
#
#     # save the model weights
#     torch.save(model.state_dict(), PATH + 'model003')
#
#     # print some output statistics
#     # sys.stdout.write("\n")
#
#
# if __name__ == "__main__":
#     main()
