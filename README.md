# OpenCL NeuralNetwork
This project demonstrates the use of an OpenCL kernel for running neural networks. Computing using an OpenCL kernel allows networks to be run on either the GPU or CPU and takes full advantage of any parallel capabilities. The neural network code supports multilayer perceptron neural nets with an arbitrary number of layers each with an arbitrary number of perceptrons. Learning is done using back-propagation with momentum. Both hyperbolic tangent and logistic activation functions are supported. The simple recurrent network architectures Elman Networks and Jordan Networks are also supported. This demo is currently setup to train and run a network learning the XOR function but can be extended to perform other tasks easily.

# Compilation & Running
These instructions have only been tested on OS X systems  
````
    $ gcc -framework opencl -lm main.c NeuralNetwork.c -o NNExample    
    $ ./NNExample
````
