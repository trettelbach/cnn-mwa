import numpy as np
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()



def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)



#
# import copy
# import cv2
#
# def preprocess_image(cv2im, resize_im=True):
#     """
#         Processes image for CNNs
#     Args:
#         PIL_img (PIL_img): Image to process
#         resize_im (bool): Resize to 224 or not
#     returns:
#         im_as_var (Pytorch variable): Variable that contains processed float tensor
#     """
#     # mean and std list for channels (Imagenet)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     # Resize image
#     if resize_im:
#         cv2im = cv2.resize(cv2im, (224, 224))
#     im_as_arr = np.float32(cv2im)
#     im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
#     im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
#     # Normalize the channels
#     for channel, _ in enumerate(im_as_arr):
#         im_as_arr[channel] /= 255
#         im_as_arr[channel] -= mean[channel]
#         im_as_arr[channel] /= std[channel]
#     # Convert to float tensor
#     im_as_ten = torch.from_numpy(im_as_arr).float()
#     # Add one more channel to the beginning. Tensor shape = 1,3,224,224
#     im_as_ten.unsqueeze_(0)
#     # Convert to Pytorch variable
#     im_as_var = Variable(im_as_ten, requires_grad=True)
#     return im_as_var
#
#
# def recreate_image(im_as_var):
#     """
#         Recreates images from a torch variable, sort of reverse preprocessing
#     Args:
#         im_as_var (torch variable): Image to recreate
#     returns:
#         recreated_im (numpy arr): Recreated image in array
#     """
#     reverse_mean = [-0.485, -0.456, -0.406]
#     reverse_std = [1/0.229, 1/0.224, 1/0.225]
#     recreated_im = copy.copy(im_as_var.data.numpy()[0])
#     for c in range(3):
#         recreated_im[c] /= reverse_std[c]
#         recreated_im[c] -= reverse_mean[c]
#     recreated_im[recreated_im > 1] = 1
#     recreated_im[recreated_im < 0] = 0
#     recreated_im = np.round(recreated_im * 255)
#
#     recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
#     # Convert RBG to GBR
#     recreated_im = recreated_im[..., ::-1]
#     return recreated_im
#
#
# class CNNLayerVisualization():
#     """
#         Produces an image that minimizes the loss of a convolution
#         operation for a specific layer and filter
#     """
#     def __init__(self, model, selected_layer, selected_filter):
#         self.model = model
#         self.model.eval()
#         self.selected_layer = selected_layer
#         self.selected_filter = selected_filter
#         self.conv_output = 0
#         # Generate a random image
#         self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
#         # Create the folder to export images if not exists
#         if not os.path.exists('../generated'):
#             os.makedirs('../generated')
#
#     def hook_layer(self):
#         def hook_function(module, grad_in, grad_out):
#             # Gets the conv output of the selected filter (from selected layer)
#             self.conv_output = grad_out[0, self.selected_filter]
#
#         # Hook the selected layer
#         self.model[self.selected_layer].register_forward_hook(hook_function)
#
#     def visualise_layer_with_hooks(self):
#         # Hook the selected layer
#         self.hook_layer()
#         # Process image and return variable
#         self.processed_image = preprocess_image(self.created_image)
#         # Define optimizer for the image
#         optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
#         for i in range(1, 31):
#             optimizer.zero_grad()
#             # Assign create image to a variable to move forward in the model
#             x = self.processed_image
#             for index, layer in enumerate(self.model):
#                 # Forward pass layer by layer
#                 # x is not used after this point because it is only needed to trigger
#                 # the forward hook function
#                 x = layer(x)
#                 # Only need to forward until the selected layer is reached
#                 if index == self.selected_layer:
#                     # (forward hook function triggered)
#                     break
#             # Loss function is the mean of the output of the selected layer/filter
#             # We try to minimize the mean of the output of that specific filter
#             loss = -torch.mean(self.conv_output)
#             print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
#             # Backward
#             loss.backward()
#             # Update image
#             optimizer.step()
#             # Recreate image
#             self.created_image = recreate_image(self.processed_image)
#             # Save image
#             if i % 5 == 0:
#                 cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
#                             '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
#                             self.created_image)
#
#     def visualise_layer_without_hooks(self):
#         # Process image and return variable
#         self.processed_image = preprocess_image(self.created_image)
#         # Define optimizer for the image
#         optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
#         for i in range(1, 31):
#             optimizer.zero_grad()
#             # Assign create image to a variable to move forward in the model
#             x = self.processed_image
#             for index, layer in enumerate(self.model):
#                 # Forward pass layer by layer
#                 x = layer(x)
#                 if index == self.selected_layer:
#                     # Only need to forward until the selected layer is reached
#                     # Now, x is the output of the selected layer
#                     break
#             # Here, we get the specific filter from the output of the convolution operation
#             # x is a tensor of shape 1x512x28x28.(For layer 17)
#             # So there are 512 unique filter outputs
#             # Following line selects a filter from 512 filters so self.conv_output will become
#             # a tensor of shape 28x28
#             self.conv_output = x[0, self.selected_filter]
#             # Loss function is the mean of the output of the selected layer/filter
#             # We try to minimize the mean of the output of that specific filter
#             loss = -torch.mean(self.conv_output)
#             print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
#             # Backward
#             loss.backward()
#             # Update image
#             optimizer.step()
#             # Recreate image
#             self.created_image = recreate_image(self.processed_image)
#             # Save image
#             if i % 5 == 0:
#                 cv2.imwrite('../generated/layer_vis_l' + str(self.selected_layer) +
#                             '_f' + str(self.selected_filter) + '_iter'+str(i)+'.jpg',
#                             self.created_image)
#
#
# # if __name__ == '__main__':
# cnn_layer = 17
# filter_pos = 5
# # Fully connected layer is not needed
# # pretrained_model = torch.load('/media/tabea/FIRENZE/mwa_data/model001').features
# pretrained_model = models.vgg16(pretrained=True).features
# layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)
#
#     # Layer visualization with pytorch hooks
# layer_vis.visualise_layer_with_hooks()
#
#     # Layer visualization without pytorch hooks
# # layer_vis.visualise_layer_without_hooks()
