# MNIST_CompVision

A simple Convolutional Neural Net (CNN) written in python sing tensorflow to recognize hand written digits from the MNIST dataset. This CNN contains 2 convolution layers followed by a fully connected layer. I also used neuron dropout to reduce overfitting and improve generalization.

Something like this could be used in combination with pole-cart to produce a vision based reward function. Although overfitting will be a problem if this exact algorithm is used. Hopefully, just a couple convolutional layers with a similarity metric and no need for parameter optimization would be able to handle that task better this. It would also greatly improve effeciency and training time of the total system. 
