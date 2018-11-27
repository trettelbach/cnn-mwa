import torch
import torch.nn as nn
import astropy.io.fits as pyfits
import torch.utils.data as data
from torch.autograd import Variable
import cnn_gleam
import myreader
import time
import sklearn.metrics
import sys
import argparse


def main():
    """
    Loads the saved weights of the CnnGleam
    model to run the network algorithm on a
    validation/testing/unknown dataset.

    Fits headers will be updated to contain information
    on the classifications made by the network
    Also prints some statistics, if desired.
    """

    parser = argparse.ArgumentParser(description='Run the GleamNet for '
                                                 'Classification')
    parser.add_argument('path', metavar='PATH', type=str, nargs='?',
                        help='path to the csv with file locations')
    parser.add_argument('statistics', metavar='STATS', type=int, nargs='?',
                        help='determine if statistics should be calculated')
    args = parser.parse_args()

    # set path to data, number of dataset labels, version number of network run
    PATH = args.path
    NLABEL = 3
    VER = 1.0

    start_time = time.time()

    # check if GPU available, and set device accordingly
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # do I want statistics?
    if args.statistics == 0:
        stats = False
        sys.stdout.write('updating fits headers only\n')
    else:
        stats = True
        sys.stdout.write('updating fits headers and producing further '
                         'statistics\n')

    # load testing set, data augmentation
    fits_test = myreader.MyCustomDatasetFits(
        PATH + 'mwa_data/archiv/pytorch_container/labels_train.csv',
        transformation='test')
    testloader = torch.utils.data.DataLoader(fits_test,
                                             shuffle=True,
                                             num_workers=1)

    # initialize the architecture for the model
    model = cnn_gleam.CnnGleam(NLABEL).to(device)
    # load saved weights to the model
    model.load_state_dict(torch.load(PATH + 'gleamnet'))

    # define criterion
    criterion = nn.CrossEntropyLoss()

    # prepare some lists for statistical evaluation
    if stats:
        test_losses = []
        test_true = []
        test_pred = []
        test_cert = []
        test_f1 = []
        test_loc = []

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
            correct_test += (max_indices == labels).sum()

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

        sys.stdout.write('correct: %d - predicted: %d at %.3f %% - image: %s\n'
                         % (labels.item(), max_indices.item(),
                            max_values.item() * 100, location))

        if stats:
            test_losses.append(test_loss.item())
            test_loc.append(location)
            test_true.append(labels.item())
            test_pred.append(max_indices.item())
            test_cert.append(max_values.item())

    if stats:
        test_acc = (100.0 * correct_test / float(len(test_loc))).item()
        test_f1.append(sklearn.metrics.f1_score(test_true,
                                                test_pred,
                                                average='micro'))

    if stats:
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

    sys.stdout.write('Duration of testing (in s): %.2f\n'
                     % (time.time() - start_time))


if __name__ == "__main__":
    main()
