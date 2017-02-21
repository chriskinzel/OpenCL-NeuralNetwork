#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "NeuralNetwork.h" 

/* 
   This example demonstrates a 3 layer 5 node 2 input feed forward multilayer neural network using the backpropagation learning rule
   that models the XOR logic function. The first layer of any neural network is always an input layer whose units only have one input 
   and do not acutally compute anything but rather transfer the value of their input to their output to act as either data compression
   or data expansion, in this case the input nodes expand the data.
*/

/* 
   XOR Truth Table
 -------------------
     In In Out
      1  1  0
      0  1  1
      1  0  1
      0  0  0
*/

/*
   Neural Network Model Diagram
 --------------------------------
		  Out
           |
           O  <- Output layer
          / \
		 O   O <- Hidden layer
		/\   /\
        \/   \/
		O     O <- Input layer
		|     |
	    In    In
*/

#define pattern_size 7

bool gpu = false;

void reccurent() {
    int numOfPerceptrons[3] =  {1,1,1};
    
    int pattern[pattern_size] = {0,1,1,0,1,1,0};
    
    float** sequence = malloc(sizeof(float)*(pattern_size-1));
    float** targets = malloc(sizeof(float)*(pattern_size-1));
    for(int i=0;i<pattern_size-1;i++) {
        sequence[i] = malloc(sizeof(float));
        targets[i] = malloc(sizeof(float));
        
        sequence[i][0] = pattern[i];
        targets[i][0] = pattern[i+1];
    }
    
    NeuralNetwork brain = CreateNeuralNetwork(3, numOfPerceptrons, 0.15, 0.1, kNetworkTypeElman, kNetworkFunctionLogistic, kNetworkLearningModeBackpropagation, gpu);
    TrainNeuralNetwork(&brain, sequence, targets, pattern_size-1, 1000000, 0.005, false);
    
    bool flag = true;
    for(int i=0;i<pattern_size-1;i++) {
        brain.inputs[0] = sequence[i][0];
        UpdateNeuralNetwork(&brain);
        
        printf("Input:%i Actual Output:%f Target:%i Output:%i\n", pattern[i], brain.outputs[0], pattern[i+1], (brain.outputs[0] > 0.5));
        
        if((brain.outputs[0] > 0.5) != pattern[i+1]) {
            flag = false;
        }
    }
    
    if(flag) {
        printf("\nMATCH\n");
    } else {
        printf("\nINCORRECT\n");
    }
    
    printf("\nMSE:%f Training Time:%f Execution Time:%f\n\n", brain.error, brain.trainingTime, brain.executionTime);
    
    ReleaseNeuralNetwork(&brain);
}

int main (int argc, const char * argv[]) {
    /*reccurent();
    gpu = true;
    reccurent();
    exit(0);*/
        
    int nodes[3] = {2,2,1};

	// Create the network
	NeuralNetwork brain = CreateNeuralNetwork(3, nodes, 0.5, 0.8, kNetworkTypeStandard, kNetworkFunctionLogistic, kNetworkLearningModeBackpropagation, gpu); // The learning rate is trial and error same with momentum high values can be used
                                                                                                                                                                              // with this simple net because it does not have to generalize at all                
                                                                                                                                                        
    float** samples = malloc(sizeof(float*) * 4);
    float** targets = malloc(sizeof(float*) * 4);
    for(int i=0;i<4;i++) {
        samples[i] = malloc(sizeof(float) * 2);
        targets[i] = malloc(sizeof(float));
    }

    samples[0][0] = -1.0;
    samples[0][1] = -1.0;
    
    targets[0][0] = 0.0;
    
    samples[1][0] = 1.0;
    samples[1][1] = -1.0;
    
    targets[1][0] = 1.0;
    
    samples[2][0] = -1.0;
    samples[2][1] = 1.0;
    
    targets[2][0] = 1.0;
    
    samples[3][0] = 1.0;
    samples[3][1] = 1.0;
    
    targets[3][0] = 0.0;
    
    TrainNeuralNetwork(&brain, samples, targets, 4, 10000, /*0.0005*/-1, false);
    
	// Output the results to determine if the network was adequately trained
	// the outputs will never actually converge to 0 or 1 completely
	brain.inputs[0] = -1.0;
	brain.inputs[1] = -1.0;
	UpdateNeuralNetwork(&brain);
	printf("Output:%f\n", brain.outputs[0]);
	
	brain.inputs[0] = 1;
	brain.inputs[1] = -1.0;
	UpdateNeuralNetwork(&brain);
	printf("Output:%f\n", brain.outputs[0]);
	
	brain.inputs[0] = -1.0;
	brain.inputs[1] = 1;
	UpdateNeuralNetwork(&brain);
	printf("Output:%f\n", brain.outputs[0]);
	
	brain.inputs[0] = 1;
	brain.inputs[1] = 1;
	UpdateNeuralNetwork(&brain);
	printf("Output:%f\n\n", brain.outputs[0]);
    	
	// When finished free the allocated memory that the network used
	ReleaseNeuralNetwork(&brain);
    
    if(gpu != true) {
        gpu = true;
        //main(argc, argv);
    }
	
    return 0;
}
