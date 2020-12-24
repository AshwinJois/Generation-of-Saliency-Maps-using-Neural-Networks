# Saliency-Maps
**Generation of Saliency Maps Using Neural Networks**

The Human eyes automatically focus on the most attractive regions. This is a naturally
occurring event. It is complicated to make the computer imitate the way the human eye
does while seeing an image. One of the methods to achieve this is by training a neural
network with a sufficient amount of dataset.
Saliency Maps define each pixel's unique quality. It is the representation of the Image in
a unique manner that is easier to analyze and is also meaningful. It is an image in the form of a map where people are expected
to look irrespective of the task they are doing. Saliency Maps are usually rendered as
a heatmap. The hotness in the heatmap refers to the region of interest, whereas the
Saliency Maps refers to the probability of each pixel.


**Arranging Dataset**

Dataset plays an essential role in training a neural network, training the neural network with a sufficient dataset leads to better learning
and better predictions. We are using the dataset available on http://saliency.mit.edu/datasets.html. The Mit Saliency Benchmark dataset used
has scenes from 20 categories. Each category consists of 200 images. In total the dataset
has 4000 images, 2000 training datasets, and 2000 sets of labels for the training dataset.
The 2000 training dataset consists of the original image having a resolution of 1920x1080
pixels. The 2000 labels are nothing but ground truth.

**Proposed Implementation**

![BlockDiagram](https://user-images.githubusercontent.com/63425115/103107126-b7775d00-463b-11eb-85f0-c5936938af5e.JPG)
The input image and its ground truth will act as the input to the neural network. The
dimensions of input image, ground truth are 512x512x3 and 512x512x1 respectively. The
enocder-decoder of the neural network processes this data and outputs a saliency map
with a dimension 512x512x1. The pre-trained model used in the project is ResNet-50. It
is not possible to achieve the desired result by using ResNet-50 alone. So, the need for
modifications in the pre-trained model is essential, we consider
the ResNet-50 as the encoder and the modifications added to it will be the decoder. As the encoder is the pre-trained model ResNet-50, it consists of 48 convolutional layers,
1 max-pooling and 1 average pooling layers. The decoder is responsible for achieving
a high resolution equivalent to the input image and also to learn features. The output of
the encoder will have a lower resolution which will be the input for the decoder.

**Training**

For the available
dataset, the bathc size of 4 is considered. The batch size is the number of samples that
will be propogated through the network at a time. The optimizer used is SGD with a
learning rate of 0.0001. The
loss used to train is Binary Crossentropy and the metric used is SSIM. SSIM is a method
used to measure the similarity between two images. SSIM measures the similarity between prediction and the ground truth image.

















