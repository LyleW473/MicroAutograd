from nn import MultiLayerPerceptron

# Guidance from: https://www.youtube.com/watch?v=VMj-3S1tku0

# Initating MLP
n = MultiLayerPerceptron(num_input = 3, num_outputs = [4, 4, 1])

# Inputs
xs = [
    [-2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ]

# Targets
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(20):

    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Setting gradients to 0 (zero grad)
    for p in n.parameters():
        p.gradient = 0.0

    # Backpropagation
    loss.backward()
    
    # Updating weights and biases
    for p in n.parameters():
        p.data += -(0.1 * p.gradient)
    
    print(k, f"Loss : {loss.data}")

# Predictions after training
print([[n(x) for x in xs]])