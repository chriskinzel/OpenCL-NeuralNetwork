//
//  NeuralNetwork Example
//
//  Created by Chris Kinzel on 11-07-19.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#ifndef NN_H
#define NN_H 

#include <OpenCL/OpenCL.h>
#include <stdbool.h>

//TODO: resilent propagation 
//TODO: try moving clSetKernelArgs (the cl_mem ones that don't change) to CreateNeuralNetwork()

#pragma mark Common

typedef enum {
    kNetworkTypeStandard = 0,
    kNetworkTypeElman,
    kNetworkTypeJordan
} NetworkType;

typedef enum {
    kNetworkFunctionLogistic = 0,
    kNetworkFunctionHyperbolicTangent
} NetworkFunction;

typedef enum {
    kNetworkLearningModeBackpropagation = 0,
    kNetworkLearningModeHebbian
} NetworkLearningMode;

#define KERNEL_COUNT 12

#pragma mark Perceptron

// Strucutre to organize data
typedef struct __attribute__ ((aligned (16))) { // 16 byte aligment
	float output; 
	float bias; 
	float error; 
	
	int numOfInputs;
    int reccurent;
} Perceptron;

#pragma mark PerceptronLayer

// Structure holding an array of perceptrons and a int to keep track of the count
typedef struct {
	Perceptron* perceptrons;	
	int numOfPerceptrons;
} PerceptronLayer;

#pragma mark NeuralNetwork

// Structure to hold and organize perceptron layers, inputs, outputs and desiredOutputs
typedef struct {	
	PerceptronLayer* perceptronLayers;
	int numOfPerceptronLayers;
    
    int training_flags;
	    
	NetworkType type; 
    NetworkFunction activation_function;
    NetworkLearningMode learning_mode;
	
	cl_float* desiredOutputs;
	cl_float* inputs;
    cl_float* outputs;
	
	cl_float** weights;
	cl_float** previous_deltas;
		
	cl_float learningRate;
	cl_float error;
	cl_float momentum;
	
	cl_float trainingTime;
	cl_float executionTime;

	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	
    cl_mem network_weight_buffer;
    cl_mem network_delta_buffer;
    cl_mem network_layer_buffer;
    
	cl_mem* weight_buffer;
	cl_mem* delta_buffer;
	cl_mem* layer_buffer;
    cl_mem error_buffer;
	cl_mem input_buffer;
	cl_mem output_buffer;
	cl_mem target_buffer;
	cl_mem null_buffer;
    
    cl_mem node_count_buffer;
    cl_mem connection_count_buffer;
	
	cl_program program;
	cl_kernel kernels[KERNEL_COUNT];
	
	bool train;
    bool online;
} NeuralNetwork;

// Create a neural network, note that numberOfPerceptrons is an array of size numberOfLayers specifying how many perceptrons there
// are in each layer. Note that layer 0 is always an input layer and its perceptrons are only ever allowed to have one input whose value
// gets transfered to its output in order to act as either data compression or expansion units. Neural networks created with this
// function must have at least 3 layers
NeuralNetwork CreateNeuralNetwork(int numberOfLayers, int numberOfPerceptrons[], float learningRate, float momentum, NetworkType type, NetworkFunction func, NetworkLearningMode mode, bool useGPU);

// Compute the outputs of the neural network and if specified train the network
void UpdateNeuralNetwork(NeuralNetwork* n);

// Train neural network with given training sets and stopping conditions such as mse and iterations
// set iterations to -1 if you want to train until the mse is reached and set mse to -1 if you want
// to run for the specified iterations only, the function returns the number of iterations it did or -1 if execution failed
int TrainNeuralNetwork(NeuralNetwork* n, float** sets, float** targets, int samples, int iterations, float mse, bool randomize);


// Save a neural network to be opened later with loadNet
void saveNet(NeuralNetwork* n, const char* filename);

// Create a neural network from a file created by saveNet
NeuralNetwork loadNet(const char* filename, bool useGPU);

// Free neural network when done
void ReleaseNeuralNetwork(NeuralNetwork* n);
#endif