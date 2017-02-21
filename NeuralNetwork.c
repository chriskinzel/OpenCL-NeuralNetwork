/*
 *  NeuralNetwork.c
 *  NeuralNetwork Example
 *
 *  Created by Chris Kinzel on 11-07-26.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

//TODO: linear activation function

#include "NeuralNetwork.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#pragma mark Private

// Training flags
#define TRAIN_NO_MSE 1
#define TRAIN_EXTERNAL_BUFFERS 2
#define TRAIN_NO_MSE_EXTERNAL_BUFFERS 3

// Kernel indices
#define UPDATE_LOGISTIC_KERNEL 0
#define UPDATE_TANH_KERNEL 1
#define ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL 2
#define ONLINE_TRAIN_TANH_HIDDEN_KERNEL 3
#define ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL 4
#define ONLINE_TRAIN_TANH_OUTPUT_KERNEL 5
#define COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL 6
#define COMPUTE_TANH_OUTPUT_ERROR_KERNEL 7
#define COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL 8
#define COMPUTE_TANH_HIDDEN_ERROR_KERNEL 9
#define BATCH_TRAIN_NETWORK_KERNEL 10
#define HEBBIAN_TRAIN_KERNEL 11

char* cl_get_error_string(cl_int err);
void CreatePerceptron(Perceptron* p, int numOfInputs, bool reccurent);
void CreatePerceptronLayer(PerceptronLayer* pl, int numberOfInputs, int numberOfPerceptrons, bool reccurent);
void loadKernels(NeuralNetwork* n);
void initOpenCL(NeuralNetwork* n, bool gpu);

char* cl_get_error_string(cl_int err) {
	switch (err) {
		case CL_INVALID_COMMAND_QUEUE:
			return "invalid command queue";
			break;
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "invalid global work size";
			break;
		case CL_INVALID_MIP_LEVEL:
			return "invalid mip level";
			break;
		case CL_INVALID_BUFFER_SIZE:
			return "invalid buffer size";
			break;
		case CL_INVALID_GL_OBJECT:
			return "invalid gl object";
			break;
		case CL_INVALID_OPERATION:
			return "invalid operation";
			break;
		case CL_INVALID_EVENT:
			return "invalid event";
			break;
		case CL_INVALID_EVENT_WAIT_LIST:
			return "invalid event wait list";
			break;
		case CL_INVALID_GLOBAL_OFFSET:
			return "invalid global offset";
			break;
		case CL_INVALID_WORK_ITEM_SIZE:
			return "invalid work item size";
			break;
		case CL_INVALID_WORK_GROUP_SIZE:
			return "invalid work group size";
			break;
		case CL_INVALID_WORK_DIMENSION:
			return "invalid work dimension";
			break;
		case CL_INVALID_KERNEL_ARGS:
			return "invalid kernel arguments";
			break;
		case CL_INVALID_ARG_SIZE:
			return "invalid argument size";
			break;
		case CL_INVALID_ARG_VALUE:
			return "invalid argument value";
			break;
		case CL_INVALID_ARG_INDEX:
			return "invalid argument index";
			break;
		case CL_INVALID_KERNEL:
			return "invalid kernel";
			break;
		case CL_INVALID_KERNEL_DEFINITION:
			return "invalid kernel definition";
			break;
		case CL_INVALID_KERNEL_NAME:
			return "invalid kernel name";
			break;
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "invalid program executable";
			break;
		case CL_INVALID_PROGRAM:
			return "invalid program";
			break;
		case CL_INVALID_BUILD_OPTIONS:
			return "invalid build options";
			break;
		case CL_INVALID_BINARY:
			return "invalid binary";
			break;
		case CL_INVALID_SAMPLER:
			return "invalid sampler";
			break;
		case CL_INVALID_IMAGE_SIZE:
			return "invalid image size";
			break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "invalid image format descriptor";
			break;
		case CL_INVALID_MEM_OBJECT:
			return "invalid memory object";
			break;
		case CL_INVALID_HOST_PTR:
			return "invalid host pointer";
			break;
		case CL_INVALID_QUEUE_PROPERTIES:
			return "invalid queue properties";
			break;
		case CL_INVALID_CONTEXT:
			return "invalid context";
			break;
		case CL_INVALID_DEVICE:
			return "invalid device";
			break;
		case CL_INVALID_PLATFORM:
			return "invalid platform";
			break;
		case CL_INVALID_DEVICE_TYPE:
			return "invalid device type";
			break;
		case CL_INVALID_VALUE:
			return "invalid value";
			break;
		case CL_MAP_FAILURE:
			return "map failure";
			break;
		case CL_BUILD_PROGRAM_FAILURE:
			return "build program failure";
			break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "image format not supported";
			break;
		case CL_IMAGE_FORMAT_MISMATCH:
			return "image format mismatch";
			break;
		case CL_MEM_COPY_OVERLAP:
			return "memory copy overlaped";
			break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "profiling info unavailable";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			return "out of host memory";
			break;
		case CL_OUT_OF_RESOURCES:
			return "out of resources";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "memory object allocation failure";
			break;
		case CL_COMPILER_NOT_AVAILABLE:
			return "compiler unavailable";
			break;
		case CL_DEVICE_NOT_AVAILABLE:
			return "device unavailable";
			break;
		case CL_DEVICE_NOT_FOUND:
			return "device not found";
			break;
		default:
			break;
	}
	
	return "nil";
}

#pragma mark Perceptron

// Create a single perceptron with numOfInputs
void CreatePerceptron(Perceptron* p, int numOfInputs, bool reccurent) {
    if(reccurent) {
        numOfInputs++;
    }
    
	p->error = 0.0;
	p->bias = 0.0;
	p->numOfInputs = numOfInputs;
    p->reccurent = reccurent;
}

#pragma mark PerceptronLayer

// Create a layer with numberOfInputs per perceptron and with perceptron count numberOfPerceptrons
void CreatePerceptronLayer(PerceptronLayer* pl, int numberOfInputs, int numberOfPerceptrons, bool reccurent) {
	pl->perceptrons = (Perceptron*)malloc(sizeof(Perceptron) * numberOfPerceptrons);
	
	for(int i=0;i<numberOfPerceptrons;i++) {
		CreatePerceptron(&pl->perceptrons[i], numberOfInputs, reccurent);
	}
	
	pl->numOfPerceptrons = numberOfPerceptrons;
}


#pragma mark NeuralNetwork

// Loads OpenCL kernels called internally don't touch
void loadKernels(NeuralNetwork* n) {
	const char* filename = "ann_kernels.cl";
	FILE* f = fopen(filename, "r");
	if(f == NULL) {
		printf("Neural Network Fatal Error - Could not locate Kernel file quitting application\n");
		abort();
	}
	
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);
	
	char* source = (char*)malloc(size);
	fread(source, size, 1, f);
	fclose(f);
    
	cl_int err;
	n->program = clCreateProgramWithSource(n->context, 1, (const char**)&source, NULL, &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Could not load kernels quitting application\n");
		abort();
	}
	
	printf("Neural Network Info - Compiling kernels...\n");
	
	err = clBuildProgram(n->program, 0, NULL, NULL, NULL, NULL);
	
	char buffer[2048];
	clGetProgramBuildInfo(n->program, n->device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
	printf("\n%s\n\n", buffer);
	
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Did not succesfully compile kernels compiler returned error code %i quitting application\n", err);
		abort();
	}
	
	printf("KERNEL COMPILATION SUCCESS\n\n");
	printf("Neural Network Info - Loading kernels...\n");
	
	n->kernels[UPDATE_LOGISTIC_KERNEL] = clCreateKernel(n->program, "update_logistic", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", UPDATE_LOGISTIC_KERNEL);
		abort();
	}
	n->kernels[UPDATE_TANH_KERNEL] = clCreateKernel(n->program, "update_tanh", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", UPDATE_TANH_KERNEL);
		abort();
	}
    n->kernels[BATCH_TRAIN_NETWORK_KERNEL] = clCreateKernel(n->program, "batch_train_network", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", BATCH_TRAIN_NETWORK_KERNEL);
		abort();
	}
    n->kernels[HEBBIAN_TRAIN_KERNEL] = clCreateKernel(n->program, "hebbian_train", &err);
    if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", HEBBIAN_TRAIN_KERNEL);
		abort();
	}
    n->kernels[COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL] = clCreateKernel(n->program, "compute_logistic_hidden_error", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL);
		abort();
	}
    n->kernels[COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL] = clCreateKernel(n->program, "compute_logistic_output_error", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL);
		abort();
	}
    n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL] = clCreateKernel(n->program, "online_train_logistic_hidden_layer", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL);
		abort();
	}
    n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL] = clCreateKernel(n->program, "online_train_logistic_output_layer", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL);
		abort();
	}
    n->kernels[COMPUTE_TANH_HIDDEN_ERROR_KERNEL] = clCreateKernel(n->program, "compute_tanh_hidden_error", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", COMPUTE_TANH_HIDDEN_ERROR_KERNEL);
		abort();
	}
    n->kernels[COMPUTE_TANH_OUTPUT_ERROR_KERNEL] = clCreateKernel(n->program, "compute_tanh_output_error", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", COMPUTE_TANH_OUTPUT_ERROR_KERNEL);
		abort();
	}
    n->kernels[ONLINE_TRAIN_TANH_HIDDEN_KERNEL] = clCreateKernel(n->program, "online_train_tanh_hidden_layer", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", ONLINE_TRAIN_TANH_HIDDEN_KERNEL);
		abort();
	}
    n->kernels[ONLINE_TRAIN_TANH_OUTPUT_KERNEL] = clCreateKernel(n->program, "online_train_tanh_output_layer", &err);
	if(err != CL_SUCCESS) {
		printf("Neural Network Fatal Error - Failed to create kernel %i quitting application\n", ONLINE_TRAIN_TANH_OUTPUT_KERNEL);
		abort();
	}
	
	printf("\nLOADED KERNELS\n\n");
}

// Sets up OpenCL called internally don't touch
void initOpenCL(NeuralNetwork* n, bool gpu) {
	cl_int err;
		
	if(gpu) {
		printf("Neural Network Info - Searching For GPU...\n");
		err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &n->device, NULL);
		if(err == 0) {
			printf("\nNeural Network Info - Found GPU");
		} else {
            printf("\nNeural Network Warning - No availble GPU\n\n Defaulting to CPU");
        }
	} else {
		printf("Neural Network Info - Searching For CPU...\n");
		err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &n->device, NULL);
		if(err == 0) {
			printf("\nNeural Network Info - Found CPU");
		} else {
            printf("Neural Network Fatal Error - No availble CPU terminating program\n");
            abort();
        }
	}
	
	n->context = clCreateContext(0, 1, &n->device, NULL, NULL, &err);
	if(err != CL_SUCCESS) {
		printf("\nNeural Network Error - Could not create OpenCL context\n");
	}
	
	char info[1024];
	char vendor[256];
	char name[256];
	size_t group_size = 0;
	size_t work_size[3] = {0, 0, 0};
	cl_ulong mem = 0;
	cl_ulong local = 0;
	cl_uint ghz = 0;
	cl_uint units = 0;
	cl_uint support = 0;
	
	clGetDeviceInfo(n->device, CL_DEVICE_EXTENSIONS, sizeof(info), info, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &ghz, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &support, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &group_size, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_size), work_size, NULL);
	clGetDeviceInfo(n->device, CL_DEVICE_NAME, sizeof(name), name, NULL);
		
	int processors = 8;
	if(group_size == 1) {
		processors = 1;
	}
		
	printf(": %s\n\n Device:%s %s\n Global Memory Size:%liMB\n Local Memory Size:%liKB\n Clock Frequency:%.2fGHz\n Cores:%i\n Processors:%i\n", info, vendor, name, (long int)mem / 1024 / 1024, (long int)local / 1024, (float)ghz / 1000, (int)units, (int)units*processors);
	
	if(support == 0) {
		printf(" Precision:float32\n");
	} else {
		printf(" Precision:double64\n");
	}
	
	printf(" Max Work Group Size:%i\n", (int)group_size);
	printf(" Max Work Item Size:%i,%i,%i\n\n", (int)work_size[0], (int)work_size[1], (int)work_size[2]);
	
	n->queue = clCreateCommandQueue(n->context, n->device, 0, NULL);
	loadKernels(n);
}

// Create a neural network, note that numberOfPerceptrons is an array of size numberOfLayers specifying how many perceptrons there
// are in each layer. Note that layer 0 is always an input layer and its perceptrons are only ever allowed to have one input whose value
// gets transfered to its output in order to act as either data compression or expansion units. Neural networks created with this
// function must have at least 3 layers

NeuralNetwork CreateNeuralNetwork(int numberOfLayers, int numberOfPerceptrons[], float learningRate, float momentum, NetworkType type, NetworkFunction func, NetworkLearningMode mode, bool useGPU) {
	NeuralNetwork n;
    
    if(numberOfLayers < 3) {
		printf("Neural Network Error - Networks must have at least 3 layers\n");
		return n;
	}
    
    int numOfInputs[numberOfLayers];
    numOfInputs[0] = 1;
    
    for(int i=1;i<numberOfLayers;i++) {
        numOfInputs[i] = numberOfPerceptrons[i-1];
    }
    
	initOpenCL(&n, useGPU);
	
    n.activation_function = func;
    n.type = type;
    n.learning_mode = mode;
    
    n.training_flags = 0;
	
	n.desiredOutputs = (cl_float*)malloc(sizeof(cl_float) * numberOfPerceptrons[numberOfLayers-1]);
	n.outputs = (cl_float*)malloc(sizeof(cl_float) * numberOfPerceptrons[numberOfLayers-1]);
	n.inputs = (cl_float*)malloc(sizeof(cl_float) * numberOfPerceptrons[0]);
	
	n.learningRate = learningRate;
	n.momentum = momentum;
	
	n.train = false;
    n.online = false;
	
	n.error = 1;
	n.trainingTime = 0;
	
	memset(n.desiredOutputs, 0, sizeof(cl_float)*numberOfPerceptrons[numberOfLayers-1]);
	memset(n.outputs, 0, sizeof(cl_float)*numberOfPerceptrons[numberOfLayers-1]);
	memset(n.inputs, 0, sizeof(cl_float)*(numOfInputs[0]*numberOfPerceptrons[0]));
	
	n.perceptronLayers = (PerceptronLayer*)malloc(sizeof(PerceptronLayer) * numberOfLayers);
	n.weights = (cl_float**)malloc(sizeof(cl_float*)*(numberOfLayers-1));
	n.previous_deltas = (cl_float**)malloc(sizeof(cl_float*)*(numberOfLayers-1));
	
	srand((unsigned int)time(NULL));
	for(int i=0;i<numberOfLayers;i++) {
        if((i > 0 && i < numberOfLayers-1 && n.type == kNetworkTypeElman) || (i == numberOfLayers-1 && n.type == kNetworkTypeJordan)) {
            CreatePerceptronLayer(&n.perceptronLayers[i], numOfInputs[i], numberOfPerceptrons[i], true);
        } else {
            CreatePerceptronLayer(&n.perceptronLayers[i], numOfInputs[i], numberOfPerceptrons[i], false);
        }
		
		if(i > 0) {
			n.weights[i-1] = (cl_float*)malloc(sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons);
			memset(n.weights[i-1], 0, sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons);
			
            for(int j=0;j<n.perceptronLayers[i].numOfPerceptrons;j++) {
                n.weights[i-1][j*n.perceptronLayers[i].perceptrons[0].numOfInputs] = 1.0f;
            }
			
			n.previous_deltas[i-1] = (cl_float*)malloc(sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons);
			memset(n.previous_deltas[i-1], 0, sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons);
		}
	}
	n.numOfPerceptronLayers = numberOfLayers;
	
	// Setup OpenCL buffers
    int total_nodes = 0;
    int total_connections = 0;
    for(int i=1;i<numberOfLayers;i++) {
        total_connections += n.perceptronLayers[i].numOfPerceptrons * n.perceptronLayers[i].perceptrons[0].numOfInputs;
        total_nodes += n.perceptronLayers[i].numOfPerceptrons;
    }
    n.network_layer_buffer = clCreateBuffer(n.context, CL_MEM_READ_WRITE, sizeof(Perceptron)*total_nodes, NULL, NULL);
    n.network_weight_buffer = clCreateBuffer(n.context, CL_MEM_READ_WRITE, sizeof(cl_float)*total_connections, NULL, NULL);
    n.network_delta_buffer = clCreateBuffer(n.context, CL_MEM_READ_WRITE, sizeof(cl_float)*total_connections, NULL, NULL);
    
	n.input_buffer = clCreateBuffer(n.context, CL_MEM_READ_WRITE, sizeof(cl_float)*(numOfInputs[0] * numberOfPerceptrons[0]), NULL, NULL);
	n.output_buffer = clCreateBuffer(n.context, CL_MEM_READ_WRITE, sizeof(cl_float)*numberOfPerceptrons[numberOfLayers-1], NULL, NULL);
	n.target_buffer = clCreateBuffer(n.context, CL_MEM_READ_ONLY, sizeof(cl_float)*numberOfPerceptrons[numberOfLayers-1], NULL, NULL);
	n.error_buffer = clCreateBuffer(n.context, CL_MEM_READ_WRITE, sizeof(cl_float)*numberOfPerceptrons[numberOfLayers-1], NULL, NULL);
    n.node_count_buffer = clCreateBuffer(n.context, CL_MEM_READ_ONLY, sizeof(cl_int)*(numberOfLayers-1), NULL, NULL);
    n.connection_count_buffer = clCreateBuffer(n.context, CL_MEM_READ_ONLY, sizeof(cl_int)*(numberOfLayers-1), NULL, NULL);
    n.null_buffer = clCreateBuffer(n.context, CL_MEM_READ_ONLY, sizeof(Perceptron), NULL, NULL);
    
	n.layer_buffer = (cl_mem*)malloc(sizeof(cl_mem)*(numberOfLayers-1));
	n.weight_buffer = (cl_mem*)malloc(sizeof(cl_mem)*(numberOfLayers-1));
	n.delta_buffer = (cl_mem*)malloc(sizeof(cl_mem)*(numberOfLayers-1));
    
    int* nodes_count = malloc(sizeof(int) * (numberOfLayers-1));
    int* connections_count = malloc(sizeof(int) * (numberOfLayers-1));
    
	cl_int err; 
    
    cl_buffer_region layer_region = {0,0};
    cl_buffer_region weight_region = {0,0};
    cl_buffer_region delta_region = {0,0};
    
	for(int i=1;i<numberOfLayers;i++) {
        layer_region.origin += layer_region.size;
        weight_region.origin += weight_region.size;
        delta_region.origin += delta_region.size;
        
        layer_region.size = sizeof(Perceptron)*n.perceptronLayers[i].numOfPerceptrons;
        weight_region.size = sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons;
        delta_region.size = sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons;
        
        // Bug in OpenCL requires me to pass an err cl_int* to clCreateSubBuffer() if i pass NULL it crashes
        n.layer_buffer[i-1] = clCreateSubBuffer(n.network_layer_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &layer_region, &err);
		n.weight_buffer[i-1] = clCreateSubBuffer(n.network_weight_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &weight_region, &err);
		n.delta_buffer[i-1] = clCreateSubBuffer(n.network_delta_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &delta_region, &err);
        
		err = clEnqueueWriteBuffer(n.queue, n.layer_buffer[i-1], CL_FALSE, 0, sizeof(Perceptron)*n.perceptronLayers[i].numOfPerceptrons, (void*)n.perceptronLayers[i].perceptrons, 0, NULL, NULL);
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - layer buffer %i writing failed %s terminating function\n", i, cl_get_error_string(err));
			return n;
		}
		err = clEnqueueWriteBuffer(n.queue, n.weight_buffer[i-1], CL_FALSE, 0, sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons, (void*)n.weights[i-1], 0, NULL, NULL);
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - weight buffer %i writing failed %s terminating function\n", i, cl_get_error_string(err));
			return n;
		}
        err = clEnqueueWriteBuffer(n.queue, n.delta_buffer[i-1], CL_FALSE, 0, sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons, (void*)n.previous_deltas[i-1], 0, NULL, NULL);
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - delta buffer %i writing failed %s terminating function\n", i, cl_get_error_string(err));
			return n;
		}
        
        nodes_count[i-1] = n.perceptronLayers[i].numOfPerceptrons;
        connections_count[i-1] = n.perceptronLayers[i].perceptrons[0].numOfInputs;
    }
    
    err = clEnqueueWriteBuffer(n.queue, n.node_count_buffer, CL_FALSE, 0, sizeof(int)*(numberOfLayers-1), (void*)nodes_count, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Neural Network Error - node count buffer writing failed %s terminating function\n", cl_get_error_string(err));
        return n;
    }
    err = clEnqueueWriteBuffer(n.queue, n.connection_count_buffer, CL_FALSE, 0, sizeof(int)*(numberOfLayers-1), (void*)connections_count, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Neural Network Error - connection count buffer writing failed %s terminating function\n", cl_get_error_string(err));
        return n;
    }
    
    clFinish(n.queue); // Wait for writes to finish
    
    return n;
}

// Compute the outputs of the neural network and if specified train the network
void UpdateNeuralNetwork(NeuralNetwork* n) {
	clock_t start = clock();
	cl_int err;
	
    if(n->training_flags < 2) {
        err = clEnqueueWriteBuffer(n->queue, n->input_buffer, CL_FALSE, 0, sizeof(cl_float)*n->perceptronLayers[0].numOfPerceptrons, (void*)n->inputs, 0, NULL, NULL);	
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - input buffer writing failed %s terminating function\n", cl_get_error_string(err));
            return;
        }
    }
	
	err = clSetKernelArg(n->kernels[n->activation_function], 0, sizeof(cl_mem), &n->input_buffer);
	err += clSetKernelArg(n->kernels[n->activation_function], 5, sizeof(cl_mem), &n->output_buffer);
	if(err != CL_SUCCESS) {
		printf("Neural Network Error - update kernel args failed terminating function\n");
		return;
	}

	// Layers are the ONLY serial steps in neural networks and thus must be run independently
	for(int i=1;i<n->numOfPerceptronLayers;i++) {
		// Each node and input is indepent of another in the same layer
		err = clSetKernelArg(n->kernels[n->activation_function], 2, sizeof(cl_mem), &n->layer_buffer[i-1]); // i
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - update kernel args 2 failed terminating function\n");
			return;
		}
		
		err = clSetKernelArg(n->kernels[n->activation_function], 4, sizeof(cl_mem), &n->weight_buffer[i-1]); // i
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - update kernel args 4 failed terminating function\n");
			return;
		}
        
		if(i == 1) {
			err = clSetKernelArg(n->kernels[n->activation_function], 1, sizeof(cl_mem), &n->null_buffer);
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - update kernel args 1 failed terminating function\n");
				return;
			}
			err = clSetKernelArg(n->kernels[n->activation_function], 3, sizeof(cl_mem), &n->layer_buffer[i]); // i + 1
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - update kernel args 3 failed terminating function\n");
				return;
			}
		} else if(i == n->numOfPerceptronLayers-1) {
			err = clSetKernelArg(n->kernels[n->activation_function], 1, sizeof(cl_mem), &n->layer_buffer[i-2]); // i - 1
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - update kernel args 1 failed terminating function\n");
				return;
			}
			err = clSetKernelArg(n->kernels[n->activation_function], 3, sizeof(cl_mem), &n->null_buffer);
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - update kernel args 3 failed terminating function\n");
				return;
			}
		} else {
			err = clSetKernelArg(n->kernels[n->activation_function], 1, sizeof(cl_mem), &n->layer_buffer[i-2]); // i - 1
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - update kernel args 1 failed terminating function\n");
				return;
			}
			err = clSetKernelArg(n->kernels[n->activation_function], 3, sizeof(cl_mem), &n->layer_buffer[i]); // i + 1
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - update kernel args 3 failed terminating function\n");
				return;
			}
		}
		
		size_t global_work_size = n->perceptronLayers[i].numOfPerceptrons;
		err = clEnqueueNDRangeKernel(n->queue, n->kernels[n->activation_function], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - update kernel layer %i returned %s failed execution terminating function\n", i, cl_get_error_string(err));
			return;
		}
	}
    
    if(n->learning_mode == kNetworkLearningModeHebbian) {
        n->executionTime = (clock()-(cl_float)start) / CLOCKS_PER_SEC;
    }
	
	if(n->train && n->learning_mode != kNetworkLearningModeHebbian && n->learningRate != 0) {
        if(n->training_flags < 2) {
            err = clEnqueueWriteBuffer(n->queue, n->target_buffer, CL_FALSE, 0, sizeof(cl_float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)n->desiredOutputs, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - target buffer write failed %s terminating function\n", cl_get_error_string(err));
                return;
            }
        }
        
        if(n->online) {
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 0, sizeof(cl_mem), &n->target_buffer); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 0 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 1, sizeof(cl_mem), &n->output_buffer); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 1 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 2, sizeof(cl_mem), &n->weight_buffer[n->numOfPerceptronLayers-2]); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 2 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 3, sizeof(cl_mem), &n->delta_buffer[n->numOfPerceptronLayers-2]); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 3 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 4, sizeof(cl_mem), &n->layer_buffer[n->numOfPerceptronLayers-2]); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 4 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 5, sizeof(cl_mem), &n->layer_buffer[n->numOfPerceptronLayers-3]); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 5 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 6, sizeof(cl_mem), &n->error_buffer); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 6 failed terminating function\n");
				return;
			} 
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 7, sizeof(float), &n->learningRate); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 7 failed terminating function\n");
				return;
			}
            err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 8, sizeof(float), &n->momentum); 
			if(err != CL_SUCCESS) {
				printf("Neural Network Error - online train kernel output layer args 8 failed terminating function\n");
				return;
			}
            
            size_t global_work_size = n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons;
            err = clEnqueueNDRangeKernel(n->queue, n->kernels[ONLINE_TRAIN_LOGISTIC_OUTPUT_KERNEL+n->activation_function], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - online train kernel output layer returned %s failed execution terminating function\n", cl_get_error_string(err));
                return;
            }
            
            // Hidden layers
            for(int i=n->numOfPerceptronLayers-2;i>=1;i--) {
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 0, sizeof(cl_mem), &n->input_buffer); 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 0 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 1, sizeof(cl_mem), &n->weight_buffer[i]); // i + 1
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 1 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 2, sizeof(cl_mem), &n->layer_buffer[i]); // i + 1
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 2 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 3, sizeof(cl_mem), &n->layer_buffer[i-1]); // i 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 3 failed terminating function\n");
                    return;
                }
                if(i == 1) {
                    err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 4, sizeof(cl_mem), &n->null_buffer); 
                    if(err != CL_SUCCESS) {
                        printf("Neural Network Error - online train kernel hidden layer args 4 failed terminating function\n");
                        return;
                    }
                } else {
                    err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 4, sizeof(cl_mem), &n->layer_buffer[i-2]); // i - 1
                    if(err != CL_SUCCESS) {
                        printf("Neural Network Error - online train kernel hidden layer args 4 failed terminating function\n");
                        return;
                    } 
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 5, sizeof(cl_mem), &n->weight_buffer[i-1]); // i
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 5 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 6, sizeof(cl_mem), &n->delta_buffer[i-1]); // i 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 6 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 7, sizeof(float), &n->learningRate); 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 7 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 8, sizeof(float), &n->momentum); 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 8 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 9, sizeof(int), &n->perceptronLayers[i+1].numOfPerceptrons); 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer args 9 failed terminating function\n");
                    return;
                }
                
                global_work_size = n->perceptronLayers[i].numOfPerceptrons;
                err = clEnqueueNDRangeKernel(n->queue, n->kernels[ONLINE_TRAIN_LOGISTIC_HIDDEN_KERNEL+n->activation_function], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - online train kernel hidden layer returned %s failed execution terminating function\n", cl_get_error_string(err));
                    return;
                }
            }
        } else {
            err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL+n->activation_function], 0, sizeof(cl_mem), &n->error_buffer); 
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - output error kernel args 0 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL+n->activation_function], 1, sizeof(cl_mem), &n->target_buffer); 
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - output error kernel args 1 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL+n->activation_function], 2, sizeof(cl_mem), &n->output_buffer); 
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - output error kernel args 2 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL+n->activation_function], 3, sizeof(cl_mem), &n->layer_buffer[n->numOfPerceptronLayers-2]); 
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - output error kernel args 3 failed terminating function\n");
                return;
            }
            
            size_t global_work_size = n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons;
            err = clEnqueueNDRangeKernel(n->queue, n->kernels[COMPUTE_LOGISTIC_OUTPUT_ERROR_KERNEL+n->activation_function], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - output error kernel returned %s failed execution terminating function\n", cl_get_error_string(err));
                return;
            }
            
            // Hidden layers
            for(int i=n->numOfPerceptronLayers-2;i>=1;i--) {
                err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL+n->activation_function], 0, sizeof(cl_mem), &n->layer_buffer[i-1]); // i
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - hidden layer error kernel args 0 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL+n->activation_function], 1, sizeof(cl_mem), &n->layer_buffer[i]); // i + 1
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - hidden layer error kernel args 1 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL+n->activation_function], 2, sizeof(cl_mem), &n->weight_buffer[i]); // i + 1
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - hidden layer error kernel args 2 failed terminating function\n");
                    return;
                }
                err = clSetKernelArg(n->kernels[COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL+n->activation_function], 3, sizeof(int), &n->perceptronLayers[i+1].numOfPerceptrons); 
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - hidden layer error kernel args 3 failed terminating function\n");
                    return;
                }
                
                size_t global_work_size = n->perceptronLayers[i].numOfPerceptrons;
                err = clEnqueueNDRangeKernel(n->queue, n->kernels[COMPUTE_LOGISTIC_HIDDEN_ERROR_KERNEL+n->activation_function], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
                if(err != CL_SUCCESS) {
                    printf("Neural Network Error - hidden layer error kernel returned %s failed execution terminating function\n", cl_get_error_string(err));
                    return;
                }
            }
            
            // Now that every node in the network has an error we can train them all at once
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 0, sizeof(cl_mem), &n->input_buffer);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 0 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 1, sizeof(cl_mem), &n->network_weight_buffer);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 2 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 2, sizeof(cl_mem), &n->network_delta_buffer);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 3 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 3, sizeof(cl_mem), &n->network_layer_buffer);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 4 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 4, sizeof(cl_mem), &n->node_count_buffer);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 5 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 5, sizeof(cl_mem), &n->connection_count_buffer);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 6 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 6, sizeof(float), &n->learningRate);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 7 failed terminating function\n");
                return;
            }
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 7, sizeof(float), &n->momentum);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 8 failed terminating function\n");
                return;
            }
            
            int real_layers_count = n->numOfPerceptronLayers-1;
            err = clSetKernelArg(n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 8, sizeof(int), &real_layers_count);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel args 9 failed terminating function\n");
                return;
            }
            
            global_work_size = 0;
            for(int i=1;i<n->numOfPerceptronLayers;i++) {
                global_work_size += n->perceptronLayers[i].numOfPerceptrons;
            }
            
            err = clEnqueueNDRangeKernel(n->queue, n->kernels[BATCH_TRAIN_NETWORK_KERNEL], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - batch train network kernel returned %s failed execution terminating function\n", cl_get_error_string(err));
                return;
            }
        }
        
        if(n->training_flags != 3 && n->training_flags != 1) {
            cl_float* sq_errors = (cl_float*)malloc(sizeof(cl_float) * n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons);
            err = clEnqueueReadBuffer(n->queue, n->error_buffer, CL_TRUE, 0, sizeof(cl_float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)sq_errors, 0, NULL, NULL);		
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - error buffer read failed %s terminating function\n", cl_get_error_string(err));
                return;
            }
            
            n->error = 0;
            for(int i=0;i<n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons;i++) {
                n->error += sq_errors[i];
            }
            n->error /= 2;
            
            free(sq_errors);
        }
        
        if(n->online) {
            err = clEnqueueReadBuffer(n->queue, n->output_buffer, CL_TRUE, 0, sizeof(cl_float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)n->outputs, 0, NULL, NULL);		
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - output buffer read failed %s terminating function\n", cl_get_error_string(err));
                return;
            }
        }
        
        if((n->training_flags == 3 || n->training_flags == 1) && !n->online) {
            clFinish(n->queue);
        }
        
		n->trainingTime += (clock()-(cl_float)start) / CLOCKS_PER_SEC;
    } else if(n->learning_mode == kNetworkLearningModeHebbian && n->learningRate != 0 && n->train) {
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 0, sizeof(cl_mem), &n->input_buffer);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 0 failed terminating function\n");
            return;
        }
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 1, sizeof(cl_mem), &n->network_weight_buffer);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 1 failed terminating function\n");
            return;
        }
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 2, sizeof(cl_mem), &n->network_layer_buffer);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 2 failed terminating function\n");
            return;
        }
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 3, sizeof(cl_mem), &n->node_count_buffer);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 3 failed terminating function\n");
            return;
        }
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 4, sizeof(cl_mem), &n->connection_count_buffer);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 4 failed terminating function\n");
            return;
        }
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 5, sizeof(float), &n->learningRate);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 5 failed terminating function\n");
            return;
        }
        int real_layers_count = n->numOfPerceptronLayers-1;
        err = clSetKernelArg(n->kernels[HEBBIAN_TRAIN_KERNEL], 6, sizeof(int), &real_layers_count);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel args 6 failed terminating function\n");
            return;
        }
        
        size_t global_work_size = 0;
        for(int i=1;i<n->numOfPerceptronLayers;i++) {
            global_work_size += n->perceptronLayers[i].numOfPerceptrons;
        }
        
        err = clEnqueueNDRangeKernel(n->queue, n->kernels[HEBBIAN_TRAIN_KERNEL], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - hebbian train network kernel returned %s failed execution terminating function\n", cl_get_error_string(err));
            return;
        }
        
        err = clEnqueueReadBuffer(n->queue, n->output_buffer, CL_TRUE, 0, sizeof(cl_float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)n->outputs, 0, NULL, NULL);		
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - output buffer read failed %s terminating function\n", cl_get_error_string(err));
			return;
		}
        
        n->trainingTime = (clock()-(cl_float)start) / CLOCKS_PER_SEC;
	} else {
		n->executionTime = (clock()-(cl_float)start) / CLOCKS_PER_SEC;
		
		// Grab outputs we only bother if were not training because we generally will not care about them
		// until training is over
		err = clEnqueueReadBuffer(n->queue, n->output_buffer, CL_TRUE, 0, sizeof(cl_float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)n->outputs, 0, NULL, NULL);		
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - output buffer read failed %s terminating function\n", cl_get_error_string(err));
			return;
		}
	}
}

// Faster than training through UpdateNeuralNetwork by reducing buffer writes
int TrainNeuralNetwork(NeuralNetwork* n, float** sets, float** targets, int samples, int iterations, float mse, bool randomize) {
	cl_int err;
    
    // Save the training state and make sure were training
    bool state = n->train;
    n->train = true;
    
    // Make sure total memory will fit on the device
    size_t target_buffer_mem = sizeof(float) * n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons * samples;
    size_t input_buffer_mem = sizeof(float) * n->perceptronLayers[0].numOfPerceptrons * samples;
    
    int total_nodes = 0;
    int connection_total = 0;
    for(int i=1;i<n->numOfPerceptronLayers;i++) {
        total_nodes += n->perceptronLayers[i].numOfPerceptrons;
        connection_total += n->perceptronLayers[i].perceptrons[0].numOfInputs * n->perceptronLayers[i].numOfPerceptrons;
    }
    
    size_t network_mem = (sizeof(Perceptron) * total_nodes) + (sizeof(float) * connection_total * 2) + (sizeof(float) * n->perceptronLayers[0].numOfPerceptrons) + (sizeof(float) * n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons);
    
    cl_ulong global_mem;
    clGetDeviceInfo(n->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &global_mem, NULL);
    
    char* byte_size_mem;
    char* byte_size_req;
    int mem_div;
    int req_div;
    
    if(global_mem / 1024 / 1024 >= 1024) {
        byte_size_mem = "GB";
        mem_div = 1073741824;
    } else {
        byte_size_mem = "MB";
        mem_div = 1048576;
    }
    
    if((target_buffer_mem + input_buffer_mem + network_mem) / 1024 / 1024 >= 1024) {
        byte_size_req = "GB";
        req_div = 1073741824;
    } else if((target_buffer_mem + input_buffer_mem + network_mem) / 1024 >= 1024) {
        byte_size_req = "MB";
        req_div = 1048576;
    } else if((target_buffer_mem + input_buffer_mem + network_mem) < 1024) {
        byte_size_req = "B";
        req_div = 1;
    } else {
        byte_size_req = "KB";
        req_div = 1024;
    }
    
    printf("Neural Network Info - Required Memory:%.2f%s Availble Memory:%.2f%s\n\n", (float)(target_buffer_mem + input_buffer_mem + network_mem) / req_div, byte_size_req, (float)global_mem / mem_div, byte_size_mem);
    
    if(target_buffer_mem + input_buffer_mem + network_mem > global_mem) { 
        printf("Neural Network Error - Device does not have enough memory for training samples terminating function\n");
        return -1;
    }
    
    // Write all training data and inputs to device
    cl_mem target_buffers[samples];
    cl_mem input_buffers[samples];
    int swaps[samples];
    
    for(int i=0;i<samples;i++) {
        swaps[i] = i;
        
        // Randomize samples
        if(rand() / (float)RAND_MAX > 0.5 && randomize) {
            float* sample = sets[i];
            float* target = targets[i];
            
            for(int j=0;j<samples;j++) {
                if(memcmp(targets[j], target, sizeof(float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons) != 0) {
                    sets[i] = sets[j];
                    sets[j] = sample;
                    
                    targets[i] = targets[j];
                    targets[j] = target;
                    
                    swaps[i] = j;
                    
                    break;
                }
            }
        }
        
        target_buffers[i] = clCreateBuffer(n->context, CL_MEM_READ_ONLY, sizeof(float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, NULL, NULL);
        input_buffers[i] = clCreateBuffer(n->context, CL_MEM_READ_ONLY, sizeof(float)*n->perceptronLayers[0].numOfPerceptrons, NULL, NULL);
        
        err = clEnqueueWriteBuffer(n->queue, target_buffers[i], CL_FALSE, 0, sizeof(float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)targets[i], 0, NULL, NULL);
        err += clEnqueueWriteBuffer(n->queue, input_buffers[i], CL_FALSE, 0, sizeof(float)*n->perceptronLayers[0].numOfPerceptrons, (void*)sets[i], 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            printf("Neural Network Error - Failed to write training data terminating function\n");
            
            // Clean up since these buffers are large
            for(int j=0;j<i;j++) {
                clReleaseMemObject(target_buffers[i]);
                clReleaseMemObject(input_buffers[i]);
            }
            
            return -1;
        }
        
        // Reorder the samples
        if(swaps[i] != i) {
            float* sample = sets[i];
            float* target = targets[i];
            
            sets[i] = sets[swaps[i]];
            sets[swaps[i]] = sample;
            
            targets[i] = targets[swaps[i]];
            targets[swaps[i]] = target;
        }
    }
    
    // We will need to restore these later
    cl_mem old_target_buffer = n->target_buffer;
    cl_mem old_input_buffer = n->input_buffer;
    
    n->training_flags = TRAIN_EXTERNAL_BUFFERS;
    
    int it = 0;
    
    if(mse == -1) {
        n->training_flags += TRAIN_NO_MSE;
        it = iterations;
        
        for(int i=0;i<=iterations;i++) {
            float error = 0.0f;
            
            for(int j=0;j<samples;j++) {
                n->target_buffer = target_buffers[j];
                n->input_buffer = input_buffers[j];
                UpdateNeuralNetwork(n);
                
                if(i == iterations) {
                    // Grab the error on the last iteration for each sample
                    cl_float* sq_errors = (cl_float*)malloc(sizeof(cl_float) * n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons);
                    err = clEnqueueReadBuffer(n->queue, n->error_buffer, CL_TRUE, 0, sizeof(cl_float)*n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons, (void*)sq_errors, 0, NULL, NULL);		
                    if(err != CL_SUCCESS) {
                        printf("Neural Network Error - error buffer read failed %s terminating function\n", cl_get_error_string(err));
                        return -1;
                    }
                    n->error = 0;
                    for(int i=0;i<n->perceptronLayers[n->numOfPerceptronLayers-1].numOfPerceptrons;i++) {
                        n->error += sq_errors[i];
                    }
                    n->error /= 2;
                    free(sq_errors);
                    
                    error += n->error;
                }
            }
            
            if(i == iterations) {
                error /= samples;
                n->error = error;
            }
        }
    } else if(iterations == -1) {
        do {
            float error = 0.0f;
            
            for(int j=0;j<samples;j++) {
                n->target_buffer = target_buffers[j];
                n->input_buffer = input_buffers[j];
                UpdateNeuralNetwork(n);
                
                error += n->error;
            }
            
            error /= samples;
            n->error = error;
            
            it++;
        } while(n->error > mse);
    } else {
        for(int i=0;i<iterations && n->error > mse;i++) {
            float error = 0.0f;
            
            for(int j=0;j<samples;j++) {
                n->target_buffer = target_buffers[j];
                n->input_buffer = input_buffers[j];
                UpdateNeuralNetwork(n);
                
                error += n->error;
            }
            
            error /= samples;
            n->error = error;
            
            it++;
        }
    }
    
    // Restore states
    n->training_flags = 0;
    n->train = state; 
    
    // Restore buffers
    n->target_buffer = old_target_buffer;
    n->input_buffer = old_input_buffer;
    
    // Clean up
    for(int i=0;i<samples;i++) {
        clReleaseMemObject(target_buffers[i]);
        clReleaseMemObject(input_buffers[i]);
    }
    
    // Formating
    int t_div = 1;
    char* t_string = "Seconds";
    
    if(n->trainingTime > 60) {
        t_string = "Minutes";
        t_div = 60;
    }
    if(n->trainingTime > 3600) {
        t_string = "Hours";
        t_div = 3600;
    }
    
    // Print training results
    printf("Neural Network Info - Training Completed:\nMSE:%f\nTime:%.2f %s\nIterations:%i\n\n", n->error, (float)(((int)(n->trainingTime)) / (float)t_div), t_string, it);
    
    return it;
}

// Save a neural network to be opened later with loadNet
void saveNet(NeuralNetwork* n, const char* filename) {
    FILE* fp;
    fp = fopen(filename, "wb");
    
    if(fp == NULL) {
		printf("Neural Network Error - Unable to create weight file \"%s\" terminating function\n", filename);
		return;
	}
 
    fwrite((char*)&n->type, 1, 1, fp);
    fwrite((char*)&n->learning_mode, 1, 1, fp);
    fwrite((char*)&n->activation_function, 1, 1, fp);
	fwrite(&n->numOfPerceptronLayers, sizeof(int), 1, fp);
	fwrite(&n->error, sizeof(float), 1, fp);
	
	for(int i=0;i<n->numOfPerceptronLayers;i++) {
		fwrite(&n->perceptronLayers[i].numOfPerceptrons, sizeof(int), 1, fp);
        
        if(i > 0) {
            cl_int err;
            err = clEnqueueReadBuffer(n->queue, n->weight_buffer[i-1], CL_FALSE, 0, sizeof(cl_float)*n->perceptronLayers[i].perceptrons[0].numOfInputs*n->perceptronLayers[i].numOfPerceptrons, (void*)n->weights[i-1], 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - Weight buffer read failed %s terminating function\n", cl_get_error_string(err));
                return;
            }
            err = clEnqueueReadBuffer(n->queue, n->layer_buffer[i-1], CL_FALSE, 0, sizeof(Perceptron)*n->perceptronLayers[i].numOfPerceptrons, (void*)n->perceptronLayers[i].perceptrons, 0, NULL, NULL);
            if(err != CL_SUCCESS) {
                printf("Neural Network Error - Weight buffer read failed %s terminating function\n", cl_get_error_string(err));
                return;
            }
        }
    }
    
    clFinish(n->queue); // Wait for reads to finish
    	
	for(int i=1;i<n->numOfPerceptronLayers;i++) {
		long index = 0;
        
		for(int j=0;j<n->perceptronLayers[i].numOfPerceptrons;j++) {
			fwrite(n->weights[i-1] + index, sizeof(float), n->perceptronLayers[i].perceptrons[j].numOfInputs, fp);
			fwrite(&n->perceptronLayers[i].perceptrons[j].bias, sizeof(float), 1, fp);
            
            index += n->perceptronLayers[i].perceptrons[j].numOfInputs;
        }
	}
 
    fclose(fp);
}

// Create a neural network from a file created by saveNet
NeuralNetwork loadNet(const char* filename, bool useGPU) {
	int layers;
    char type;
    char learning_mode;
    char function;
	float error;
	
	FILE* fp;
	fp = fopen(filename, "rb");
    
    if(fp == NULL) {
		printf("Neural Network Fatal Error - Unable to find weight file \"%s\" terminating program\n", filename);
		abort();
	}
	
    fread(&type, 1, 1, fp);
    fread(&learning_mode, 1, 1, fp);
    fread(&function, 1, 1, fp);
	fread(&layers, sizeof(int), 1, fp);
	fread(&error, sizeof(float), 1, fp);
	
	int* numOfNodes = (int*)malloc(sizeof(int) * layers);	
	for(int i=0;i<layers;i++) {
		fread(&numOfNodes[i], sizeof(int), 1, fp);
	}
	
    NeuralNetwork n = CreateNeuralNetwork(layers, numOfNodes, 0.0, 0.0, (int)type, (int)function, (int)learning_mode, useGPU);
	n.error = error;
	
	free(numOfNodes);
	
	for(int i=1;i<n.numOfPerceptronLayers;i++) {	
        long index = 0;
        
		for(int j=0;j<n.perceptronLayers[i].numOfPerceptrons;j++) {			
			fread(n.weights[i-1] + index, sizeof(float), n.perceptronLayers[i].perceptrons[j].numOfInputs, fp);			
			fread(&n.perceptronLayers[i].perceptrons[j].bias, sizeof(float), 1, fp);
                        
            index += n.perceptronLayers[i].perceptrons[j].numOfInputs;
		}
		
		cl_int err;
		err = clEnqueueWriteBuffer(n.queue, n.weight_buffer[i-1], CL_FALSE, 0, sizeof(cl_float)*n.perceptronLayers[i].perceptrons[0].numOfInputs*n.perceptronLayers[i].numOfPerceptrons, (void*)n.weights[i-1], 0, NULL, NULL);
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - Weight buffer writing failed %s terminating function\n", cl_get_error_string(err));
			return n;
		}
		err = clEnqueueWriteBuffer(n.queue, n.layer_buffer[i-1], CL_FALSE, 0, sizeof(Perceptron)*n.perceptronLayers[i].numOfPerceptrons, (void*)n.perceptronLayers[i].perceptrons, 0, NULL, NULL);
		if(err != CL_SUCCESS) {
			printf("Neural Network Error - Layer buffer writing failed %s terminating function\n", cl_get_error_string(err));
			return n;
		}
	}
    
    clFinish(n.queue); // Wait for writes to finish
    
    fclose(fp);	
    
    return n;
}

// Free neural network when done
void ReleaseNeuralNetwork(NeuralNetwork* n) {
	free(n->desiredOutputs);
	free(n->inputs);
	free(n->outputs);
    
    clReleaseMemObject(n->network_delta_buffer);
    clReleaseMemObject(n->network_weight_buffer);
    clReleaseMemObject(n->network_layer_buffer);
	
	clReleaseMemObject(n->input_buffer);
	clReleaseMemObject(n->output_buffer);
	clReleaseMemObject(n->target_buffer);
    clReleaseMemObject(n->error_buffer);
    clReleaseMemObject(n->node_count_buffer);
    clReleaseMemObject(n->connection_count_buffer);
	clReleaseMemObject(n->null_buffer);
	
	for(int i=1;i<n->numOfPerceptronLayers;i++) {
		clReleaseMemObject(n->layer_buffer[i-1]);
		clReleaseMemObject(n->weight_buffer[i-1]);
		clReleaseMemObject(n->delta_buffer[i-1]);
		
		free(n->perceptronLayers[i].perceptrons);
		free(n->weights[i-1]);
		free(n->previous_deltas[i-1]);
	}
	
	free(n->layer_buffer);
	free(n->weight_buffer);
	free(n->delta_buffer);
	free(n->weights);
	free(n->previous_deltas);
	free(n->perceptronLayers);
	
	clReleaseContext(n->context);
    clReleaseProgram(n->program);
    
	for(int i=0;i<KERNEL_COUNT;i++) {
		clReleaseKernel(n->kernels[i]);
	}
    
    clFinish(n->queue);
	clReleaseCommandQueue(n->queue);
}