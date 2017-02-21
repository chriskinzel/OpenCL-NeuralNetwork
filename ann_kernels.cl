// Structure to hold data that a perceptron needs to function
typedef struct __attribute__ ((aligned (16))) { // 16 byte aligment
    float output;
	float bias;
	float error;
	
	int numOfInputs;
    int reccurent;
} Perceptron;

// The logistic function has moderate training and fast execution time
// the logistic function is also hardware accelerated (if supported)
// through the use of native_exp()
__kernel void update_logistic(__global float* inputs, __global Perceptron* prev_layer, __global Perceptron* current_layer, __global Perceptron* next_layer, __global float* layer_weights, __global float* outputs) {
    size_t current_node = get_global_id(0);	
    
	float sum = current_layer[current_node].bias; // Bias 
	if(prev_layer[0].numOfInputs == 0) { // First hidden layer
		for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
            if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
                sum = mad(current_layer[current_node].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            } else {
                sum = mad(inputs[i], layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            }
		}
        
		current_layer[current_node].output = native_recip((float)(1.0 + native_exp((float)-sum)));
	} else if(next_layer[0].numOfInputs == 0) { // Output layer
		for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
            if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
                sum = mad(outputs[current_node], layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            } else {
                sum = mad(prev_layer[i].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            }
		}
		
		outputs[current_node] = native_recip((float)(1.0 + native_exp((float)-sum)));
        current_layer[current_node].output = outputs[current_node];
	} else { // Hidden layers
		for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
            if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
                sum = mad(current_layer[current_node].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            } else {
                sum = mad(prev_layer[i].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            }
		}
		
		current_layer[current_node].output = native_recip((float)(1.0 + native_exp((float)-sum)));
	}
}

// The hyberolic tangent function is faster to train than the 
// logistic function but has the slowest execution time
// tanh also converges better than logistic
__kernel void update_tanh(__global float* inputs, __global Perceptron* prev_layer, __global Perceptron* current_layer, __global Perceptron* next_layer, __global float* layer_weights, __global float* outputs) {
	size_t current_node = get_global_id(0);	
	float sum = current_layer[current_node].bias; // Bias
				
	if(prev_layer[0].numOfInputs == 0) { // First hidden layer
		for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
			if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
                sum = mad(current_layer[current_node].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            } else {
                sum = mad(inputs[i], layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            }
		}
		current_layer[current_node].output = tanh((float)sum);
	} else if(next_layer[0].numOfInputs == 0) { // Output layer
		for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
            if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
                sum = mad(outputs[current_node], layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            } else {
                sum = mad(prev_layer[i].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            }
		}
		
		outputs[current_node] = tanh((float)sum);
        current_layer[current_node].output = outputs[current_node];
	} else { // Hidden layers
		for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
            if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
                sum = mad(current_layer[current_node].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            } else {
                sum = mad(prev_layer[i].output, layer_weights[current_node*current_layer[current_node].numOfInputs+i], sum);
            }
		}
		
		current_layer[current_node].output = tanh((float)sum);
	}
}

// This kernel trains the logistic output layer online
__kernel void online_train_logistic_output_layer(__global float* targets, __global float* outputs, __global float* output_weights, __global float* prev_delta, __global Perceptron* output_layer, __global Perceptron* next_layer, __global float* errors, float learning_rate, float momentum) {
    size_t current_node = get_global_id(0);
    
    errors[current_node] = outputs[current_node] * (1.0 - outputs[current_node]) * (targets[current_node] - outputs[current_node]);
    output_layer[current_node].error = errors[current_node];
    
    float delta = errors[current_node] * learning_rate;
    errors[current_node] *= errors[current_node];
    
    for(int i=0;i<output_layer[current_node].numOfInputs;i++) {
        size_t index = current_node*output_layer[current_node].numOfInputs+i;
        
        float d;
        if(i == output_layer[current_node].numOfInputs-1 && output_layer[current_node].reccurent) {
            d = delta * outputs[current_node];
        } else {
            d = delta * next_layer[i].output; 
        }
        
        output_weights[index] += d + momentum * prev_delta[index];
        prev_delta[index] = d;
    }
    
    output_layer[current_node].bias += delta;
}

// This kernel trains logistic hidden layers online
__kernel void online_train_logistic_hidden_layer(__global float* inputs, __global float* prev_weights, __global Perceptron* prev_layer, __global Perceptron* current_layer, __global Perceptron* next_layer, __global float* layer_weights, __global float* prev_delta, float learning_rate, float momentum, int prev_node_count) {
    size_t current_node = get_global_id(0);
    
    float sum = 0.0;
    for(int i=0;i<prev_node_count;i++) {
        size_t index;
        if(prev_layer[0].reccurent) {
            index = i*(prev_layer[0].numOfInputs-1)+current_node;
        } else {
            index = i*prev_layer[0].numOfInputs+current_node; 
        }
        
        sum = mad(prev_weights[index], prev_layer[i].error, sum);
    }
    
    float error = current_layer[current_node].output * (1.0 - current_layer[current_node].output) * sum;
    current_layer[current_node].error = error;
    
    float delta = error * learning_rate;
    for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
        size_t index = current_node*current_layer[current_node].numOfInputs+i;
       
        float d;
        if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
            d = delta * current_layer[current_node].output;
        } else {
            if(next_layer[0].numOfInputs == 0) {
                d = delta * inputs[i];
            } else {
                d = delta * next_layer[i].output;
            }
        }
        
        layer_weights[index] += d + momentum * prev_delta[index];
        prev_delta[index] = d;
    }
    
    current_layer[current_node].bias += delta;
}

// This kernel trains the tanh output layer online
__kernel void online_train_tanh_output_layer(__global float* targets, __global float* outputs, __global float* output_weights, __global float* prev_delta, __global Perceptron* output_layer, __global Perceptron* next_layer, __global float* errors, float learning_rate, float momentum) {
    size_t current_node = get_global_id(0);
    
    errors[current_node] = (1.0 - outputs[current_node] * outputs[current_node]) * (targets[current_node] - outputs[current_node]);
    output_layer[current_node].error = errors[current_node];
    
    float delta = errors[current_node] * learning_rate;
    errors[current_node] *= errors[current_node];
    
    for(int i=0;i<output_layer[current_node].numOfInputs;i++) {
        size_t index = current_node*output_layer[current_node].numOfInputs+i;
        float d;
        if(i == output_layer[current_node].numOfInputs-1 && output_layer[current_node].reccurent) {
            d = delta * outputs[current_node];
        } else {
            d = delta * next_layer[i].output; 
        }
        
        output_weights[index] += d + momentum * prev_delta[index];
        prev_delta[index] = d;
    }
    
    output_layer[current_node].bias += delta;
}

// This kernel trains tanh hidden layers online
__kernel void online_train_tanh_hidden_layer(__global float* inputs, __global float* prev_weights, __global Perceptron* prev_layer, __global Perceptron* current_layer, __global Perceptron* next_layer, __global float* layer_weights, __global float* prev_delta, float learning_rate, float momentum, int prev_node_count) {
    size_t current_node = get_global_id(0);
    
    float sum = 0.0;
    for(int i=0;i<prev_node_count;i++) {
        size_t index;
        if(prev_layer[0].reccurent) {
            index = i*(prev_layer[0].numOfInputs-1)+current_node;
        } else {
            index = i*prev_layer[0].numOfInputs+current_node; 
        }
        
        sum = mad(prev_weights[index], prev_layer[i].error, sum);
    }
    
    float error = (1.0 - current_layer[current_node].output * current_layer[current_node].output) * sum;
    current_layer[current_node].error = error;
    
    float delta = error * learning_rate;
    for(int i=0;i<current_layer[current_node].numOfInputs;i++) {
        size_t index = current_node*current_layer[current_node].numOfInputs+i;
        
        float d;
        if(i == current_layer[current_node].numOfInputs-1 && current_layer[current_node].reccurent) {
            d = delta * current_layer[current_node].output;
        } else {
            if(next_layer[0].numOfInputs == 0) {
                d = delta * inputs[i];
            } else {
                d = delta * next_layer[i].output;
            }
        }
        
        layer_weights[index] += d + momentum * prev_delta[index];
        prev_delta[index] = d;
    }
    
    current_layer[current_node].bias += delta;
}

// Calculates the error of the ouput layer logistic units
__kernel void compute_logistic_output_error(__global float* errors, __global float* targets, __global float* outputs, __global Perceptron* output_layer) {
   size_t current_node = get_global_id(0);
    
    errors[current_node] = outputs[current_node] * (1.0 - outputs[current_node]) * (targets[current_node] - outputs[current_node]);
    output_layer[current_node].error = errors[current_node];
    errors[current_node] *= errors[current_node]; 
}

// Calculates the error of hidden layer logistic units
__kernel void compute_logistic_hidden_error(__global Perceptron* current_layer, __global Perceptron* prev_layer, __global float* prev_weights, int prev_node_count) {
    size_t current_node = get_global_id(0);
    
    float sum = 0.0;
    for(int i=0;i<prev_node_count;i++) {
        size_t index;
        if(prev_layer[0].reccurent) {
            index = i*(prev_layer[0].numOfInputs-1)+current_node;
        } else {
           index = i*prev_layer[0].numOfInputs+current_node; 
        }
        
        sum = mad(prev_weights[index], prev_layer[i].error, sum);
    }
    
    current_layer[current_node].error = current_layer[current_node].output * (1.0 - current_layer[current_node].output) * sum;
}

// Calculates the error of the ouput layer hyperbolic tangent units
__kernel void compute_tanh_output_error(__global float* errors, __global float* targets, __global float* outputs, __global Perceptron* output_layer) {
    size_t current_node = get_global_id(0);
    
    errors[current_node] = (1.0 - outputs[current_node] * outputs[current_node]) * (targets[current_node] - outputs[current_node]);
    output_layer[current_node].error = errors[current_node];
    errors[current_node] *= errors[current_node];    
}

// Calculates the error of hidden layer hyperbolic tangent units
__kernel void compute_tanh_hidden_error(__global Perceptron* current_layer, __global Perceptron* prev_layer, __global float* prev_weights, int prev_node_count) {
    size_t current_node = get_global_id(0);
    
    float sum = 0.0;
    for(int i=0;i<prev_node_count;i++) {
        size_t index;
        if(prev_layer[0].reccurent) {
            index = i*(prev_layer[0].numOfInputs-1)+current_node;
        } else {
            index = i*prev_layer[0].numOfInputs+current_node; 
        }
        
        sum = mad(prev_weights[index], prev_layer[i].error, sum);
    }
    
    current_layer[current_node].error = (1.0 - current_layer[current_node].output * current_layer[current_node].output) * sum;
}

// After all the errors have been calculated then this kernel trains every node
__kernel void batch_train_network(__global float* inputs, __global float* network_weights, __global float* network_prev_deltas, __global Perceptron* layers, __constant int* nodes_count, __constant int* connections_count, float learning_rate, float momentum, int layer_count) {
    size_t current_node = get_global_id(0);
    
    size_t current_layer; 
    size_t pre_index = 0;
    
    size_t node_sum = 0;
    
    for(int i=0;i<layer_count;i++) {
        node_sum += nodes_count[i];
        if(current_node < node_sum) {
            current_layer = i;
            break;
        }
        pre_index += nodes_count[i] * connections_count[i];
    }
    
    size_t layer_node;    
    if(current_layer == 0) {
        layer_node = current_node;
    } else {
        layer_node = current_node - nodes_count[current_layer-1];
    }
    
    float delta = layers[current_node].error * learning_rate;
    for(int i=0;i<layers[current_node].numOfInputs;i++) {
        size_t index = pre_index+(layer_node*layers[current_node].numOfInputs+i); 
        
        float d;
        if(i == layers[current_node].numOfInputs-1 && layers[current_node].reccurent) {
            d = delta * layers[current_node].output;
        } else {
            if(current_layer == 0) {
                d = delta * inputs[i];
            } else {
                d = delta * layers[(node_sum - nodes_count[current_layer-1] - 1)+i].output;
            }
        }
        
        network_weights[index] += d + momentum * network_prev_deltas[index];
        network_prev_deltas[index] = d;
    }
    
    layers[current_node].bias += delta;
}

__kernel void hebbian_train(__global float* inputs, __global float* network_weights, __global Perceptron* layers, __constant int* nodes_count, __constant int* connections_count, float learning_rate, int layer_count) {
    size_t current_node = get_global_id(0);
    
    size_t current_layer; 
    size_t pre_index = 0;
    
    size_t node_sum = 0;
    
    for(int i=0;i<layer_count;i++) {
        node_sum += nodes_count[i];
        if(current_node < node_sum) {
            current_layer = i;
            break;
        }
        pre_index += nodes_count[i] * connections_count[i];
    }
    
    size_t layer_node;    
    if(current_layer == 0) {
        layer_node = current_node;
    } else {
        layer_node = current_node - nodes_count[current_layer-1];
    }
    
    float delta = layers[current_node].output * learning_rate;
    for(int i=0;i<layers[current_node].numOfInputs;i++) {
        size_t index = pre_index+(layer_node*layers[current_node].numOfInputs+i); 
        
        float d;
        if(i == layers[current_node].numOfInputs-1 && layers[current_node].reccurent) {
            d = delta * layers[current_node].output;
        } else {
            if(current_layer == 0) {
                d = delta * inputs[i];
            } else {
                d = delta * layers[(node_sum - nodes_count[current_layer-1] - 1)+i].output;
            }
        }
        
        network_weights[index] += d;
    }
    
    layers[current_node].bias += delta;
}