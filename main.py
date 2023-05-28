from nn import MultiLayerPerceptron

# Guidance from: https://www.youtube.com/watch?v=VMj-3S1tku0

# Initating MLP
mlp = MultiLayerPerceptron(num_input = 3, num_outputs = [4, 4, 1])

# Inputs
xs = [
    [-2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ]

# Targets
ys = [1.0, -1.0, -1.0, 1.0]

# Train MLP
mlp.train(epochs = 500, inputs = xs, targets = ys)

# Predictions after training
predictions = [mlp(x) for x in xs]

# Finding accuracy
predictions_and_targets = list(zip([mlp(x).data for x in xs], ys)) # Bundles predictions with targets for each input list in xs
errors = [((target - prediction)/ target) for prediction, target in predictions_and_targets]
average_error = sum(errors) / len(errors)
accuracy = (1 - average_error) * 100
print(f"Accuracy: {accuracy} %")