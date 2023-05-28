from value import Value
import random

class Neuron:

    def __init__(self, num_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)] # Create a single weight between this neuron and the input for all inputs
        self.bias = Value(random.uniform(-1, 1)) # One bias for each set of weights

    def __call__(self, x):
        # w * x + b [Where (w * x) = dot product]
        # Pairs all weights of this neuron with the list of inputs
        activation = sum((wi * xi for wi, xi in zip(self.weights, x)), start = self.bias) # Same as sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
        output = activation.tanh() # Tanh = activation function to prevent linearity
        
        return output

    def parameters(self):
        return self.weights + [self.bias] # Merge weights and biases into a single list
    
class Layer:

    def __init__(self, num_input, num_output): # num_output = number of neurons in this layer
        self.neurons = [Neuron(num_inputs = num_input) for _ in range(num_output)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons] # Calls the __call__ function for all neurons in this layer
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):

        # The same as:
        # parameters = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     parameters.extend(ps)

        return [p for neuron in self.neurons for p in neuron.parameters()]

class MultiLayerPerceptron:

    def __init__(self, num_input, num_outputs): # num_outputs = List containing the number of neurons for each layer e.g. [4, 4, 1]
        sizes = [num_input] + num_outputs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(num_outputs))] 
        # Layers:
        # E.g. 3 inputs:
        # [3, 4], [4, 4], [4, 1] # Where 1st num = number of inputs for the layer, 2nd num = number of outputs for this layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # Calls the __call__ function for all layers in the MLP
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def train(self, epochs, inputs, targets):
        
        for k in range(epochs):

            # Forward pass
            predictions = [self(x) for x in inputs] # Find the predictions of the NN
            loss = sum((predic - actual) ** 2 for actual, predic in zip(targets, predictions)) # Finds the MSE (Mean Squared Error) loss for this epoch

            # Setting gradients to 0 (zero grad)
            # Note: This is to ensure that previous calculations are not influencing this epoch because the gradients are accumulated, meaning the gradients will only reflect the current batch
            for p in self.parameters():
                p.gradient = 0.0

            # Backpropagation
            loss.backward() # This will alter the gradients of all the weights and biases, referring to its influence on the loss generated
            
            # Updating weights and biases
            for p in self.parameters():
                # Adding (learning_rate * p.gradient) will maximise the loss 
                # Adding -(learning_rate * p.gradient) will minimise the loss
                p.data += -(0.1 * p.gradient) 
            
            print(f"Epoch: {k + 1}, Loss: {loss.data}")
