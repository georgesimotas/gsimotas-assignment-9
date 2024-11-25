import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # Compute hidden layer pre-activation
        self.Z1 = np.dot(X, self.W1) + self.b1

        # Apply activation function
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))

        # Compute output layer pre-activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2

        # Clip Z2 to avoid overflow
        self.Z2 = np.clip(self.Z2, -500, 500)  # Prevent overflow
        self.out = 1 / (1 + np.exp(-self.Z2))  # Sigmoid activation
        return self.out

    def backward(self, X, y):
        # Output layer error
        delta2 = self.out - y  # dL/dZ2

        # Gradients for W2 and b2
        dW2 = np.dot(self.A1.T, delta2)  # dL/dW2
        db2 = np.sum(delta2, axis=0, keepdims=True)  # dL/db2

        # Backpropagate to hidden layer
        delta1 = np.dot(delta2, self.W2.T)  # dL/dA1
        if self.activation_fn == 'tanh':
            delta1 *= (1 - np.tanh(self.Z1) ** 2)  # dA1/dZ1
        elif self.activation_fn == 'relu':
            delta1 *= (self.Z1 > 0).astype(float)  # dA1/dZ1
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.Z1))
            delta1 *= sig * (1 - sig)  # dA1/dZ1

        # Gradients for W1 and b1
        dW1 = np.dot(X.T, delta1)  # dL/dW1
        db1 = np.sum(delta1, axis=0, keepdims=True)  # dL/db1

        # Update weights and biases using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):  # Perform multiple steps per frame for smoother animations
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.A1  # Activations in the hidden layer
    if hidden_features.shape[1] == 3:  # Ensure 3 neurons in the hidden layer
        ax_hidden.scatter(
            hidden_features[:, 0],
            hidden_features[:, 1],
            hidden_features[:, 2],
            c=y.ravel(),
            cmap='bwr',
            alpha=0.7
        )
    ax_hidden.set_title(f"Hidden Space at Step {frame}")

    # Decision boundary in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame}")

    # Gradient visualization (network-style with line thickness)
    for i in range(mlp.W1.shape[0]):  # Input to Hidden connections
        for j in range(mlp.W1.shape[1]):
            ax_gradient.plot(
                [0.1 + i * 0.2, 0.4 + j * 0.2],  # x-coordinates from input to hidden
                [0.2, 0.5],                      # y-coordinates from input to hidden
                linewidth=np.abs(mlp.W1[i, j]) * 5,  # Line thickness for gradients
                color="purple",
                alpha=0.8
            )
    for j in range(mlp.W2.shape[0]):  # Hidden to Output connections
        ax_gradient.plot(
            [0.4 + j * 0.2, 0.9],  # x-coordinates from hidden to output
            [0.5, 0.8],            # y-coordinates from hidden to output
            linewidth=np.abs(mlp.W2[j, 0]) * 5,  # Line thickness for gradients
            color="blue",
            alpha=0.8
        )

    # Draw nodes for input, hidden, and output layers
    input_positions = [0.1 + i * 0.2 for i in range(mlp.W1.shape[0])]
    hidden_positions = [0.4 + j * 0.2 for j in range(mlp.W1.shape[1])]
    output_position = [0.9]

    ax_gradient.scatter(input_positions, [0.2] * len(input_positions), s=200, color="blue", label="Inputs")
    ax_gradient.scatter(hidden_positions, [0.5] * len(hidden_positions), s=200, color="blue", label="Hidden")
    ax_gradient.scatter(output_position, [0.8], s=200, color="blue", label="Output")

    ax_gradient.set_title(f"Gradients at Step {frame}")
    ax_gradient.axis("off")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
