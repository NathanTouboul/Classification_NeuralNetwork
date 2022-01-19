# Classification_neural_network

## Design, train and test a Convolutional Neural Network (CNN) for a multi-class classification problem. 

[//]: <> (Description of the project)
This project is centered on the MNIST handwritten digit database with additive correlated noise.
The images and class labels are contained in the file 'MNIST_CorrNoise.npz' (inside the directory "dataset").

![Image Caption](figures/Noisy%20Handwritten%20digits.png)
___
### Objectives

- Designing and training a convolutional neural network that classifies a noisy handwritten digit into one of the 10 possible classes (0, 1, 2, 3, 4, 5, 6, 7, 8 or 9)
- Exploring different layer combinations and training settings to find the best model in terms of classification accuracy

___ 
### Implementation

Files in this project:

- main.py: main file
- preprocess.py: Import the mnist dataset and create training and testing images and associated labels
- neural_networks.py: Generate neural network models
  - the function generate_neural_network(*args) implements a few different architectures. The architecture can be chosen inside main.py among:
    - "CNN_initial" : Neural Network 1: CNN with 1 layer
    - "CNN_add_dense" : Neural Network 2: CNN adding dense layers
    - "full_dense" : Neural Network 3: CNN adding dense layers
    - "CNN_pooling" : Neural_network 4: CNN full conv 2D and max pooling
  - Modifications of parameters: type of layers: (Dense, Conv2D, MaxPooling2D), size of kernel for convolution, number of hidden layers, optimizer...
- plotting_saving.py: contains functions to plot our results


___
### Results
This project allows us to quickly study the influence of parameters, type of layers for this dataset. It can easily output figures comparing a specific parameter (such as the optimizer used), training and testing accuracies and losses, etc..

![Image Caption](figures/Comparison%20for%20the%20choice%20of%20optimizer.png)
![Image Caption](figures/Training%20model%20CNN_pooling.png)


