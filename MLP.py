import random
import numpy as np


class MLP(object):

    def __init__(self, sizes, type_activation="sigmoid"):
        self.type_activation = type_activation
        self.number_of_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[1:], sizes[:-1])]

    def activation_function(self, z):
        if self.type_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        elif self.type_activation == "relu":
            return np.maximum(0, z)
        elif self.type_activation == "leaky_relu":
            return np.where(z > 0, z, 0.01 * z)
        elif self.type_activation == "tanh":
            return np.tanh(z)
        else:
            raise ValueError("Non-existent type of activation function")

    def activation_function_derivative(self, z):
        if self.type_activation == "sigmoid":
            return self.activation_function(z) * (1 - self.activation_function(z))
        elif self.type_activation == "relu":
            return np.where(z > 0, 1, 0)
        elif self.type_activation == "leaky_relu":
            return np.where(z > 0, 1, 0.01)
        elif self.type_activation == "tanh":
            return 1 - (np.tanh(z) ** 2)
        else:
            raise ValueError("Non-existent type of activation_function_derivative")

    def forward_propagation(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(weight, a) + bias)
        return a

    def stochastic_GD(self, training_data, epochs, batch_size, lr, test_data=None):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k + batch_size]
                for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, lr)

            if test_data:
                n_test = len(test_data)
                print(f"Epoch {i}: {int(self.to_test(test_data) / n_test * 100)} % right answers")
            else:
                print(f"Epoch {i} complete")

        # with open('weights.npy', 'wb') as f:
        #     np.save(f, self.weights)

    def update_batch(self, mini_batch, eta):
        total_shift_b = [np.zeros(b.shape) for b in self.biases]
        total_shift_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            current_shift_b, current_shift_w = self.back_propagation(x, y)
            total_shift_b = [nb + dnb for nb, dnb in zip(total_shift_b, current_shift_b)]
            total_shift_w = [nw + dnw for nw, dnw in zip(total_shift_w, current_shift_w)]

        self.weights = [w - (eta / len(mini_batch)) * tw
                        for w, tw in zip(self.weights, total_shift_w)]
        self.biases = [b - (eta / len(mini_batch)) * tb
                       for b, tb in zip(self.biases, total_shift_b)]

    def back_propagation(self, x, y):
        shift_b = [np.zeros(b.shape) for b in self.biases]
        shift_w = [np.zeros(w.shape) for w in self.weights]

        # forward_propagation
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        # backward_propagation
        delta = self.cost_derivative(activations[-1], y) * self.activation_function_derivative(zs[-1])
        shift_b[-1] = delta
        shift_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.number_of_layers):
            z = zs[-l]
            sp = self.activation_function_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            shift_b[-l] = delta
            shift_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return shift_b, shift_w

    def to_test(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y
