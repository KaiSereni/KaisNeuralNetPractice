from dataclasses import dataclass
from statistics import mean
import numpy as np
from numpy import typing as npt
from typing import Callable
import math, random

ActivationFunctionType = Callable[[npt.NDArray[np.float16]], npt.NDArray[np.float16]]

class ActivationFunctions:
    RELU: ActivationFunctionType = staticmethod(lambda z: np.maximum(z, 0))
    SIGMOID: ActivationFunctionType = staticmethod(lambda z: 1.0 / (1.0 + np.exp(-z)))

@dataclass
class Datapoint:
    classification_index: int
    independents: npt.NDArray[np.float16]

    def expected_out_array(self, items: int) -> np.ndarray:
        return np.asarray([0 if p != self.classification_index else 1 for p in range(items)])
    
@dataclass
class ClassificationTrainingData:
    data: list[Datapoint]
    _num_independents: int
    _headers: list[str]
    _classes: set[int]

    def __init__(self, file_path: str):
        if not file_path.lower().endswith('.csv'):
            raise TypeError("Input file must be a CSV!")
        with open(file_path, 'r', encoding="utf-8") as f:
            csvdata = [line for line in f]
        print("Which of these items is the classification (output)?")
        headers = [w.strip() for w in csvdata[0].split(',')]
        self._headers = headers
        self._num_independents = len(headers)-1
        for i in range(len(headers)):
            print(f"[{i+1}] - {headers[i]}")
        dependent_index = input(f'Type a number from 1 - {len(headers)}: ')
        dependent_index = int(dependent_index.strip())-1
        if not 0 <= dependent_index < len(headers):
            raise ValueError("Invalid selection!")
        self._headers.pop(dependent_index)
        csvdata.pop(0)
        self.data = []
        for line in csvdata:
            if not line.strip():
                continue
            this_line_data = [float(w.strip()) for w in line.split(',')]
            this_datapoint = Datapoint(
                classification_index=int(this_line_data[dependent_index]),
                independents=np.asarray([this_line_data[i] for i in range(len(this_line_data)) if i != dependent_index]).astype(np.float16)
            )
            self.data.append(this_datapoint)
        self._classes = set([d.classification_index for d in self.data])
    
    def __str__(self):
        return "\n".join([
            f"Number of datapoints: {len(self.data)}",
            f"Independent variables per datapoint: {self._num_independents}",
            f"Class value range: [{min(self._classes)} - {max(self._classes)}]",
            f"Variable names (in order): {', '.join(self._headers)}"
        ])

class FCL_1D:
    weights: npt.NDArray[np.float16]
    biases: npt.NDArray[np.float16]
    inputs: int
    activation_function: ActivationFunctionType
    shape: tuple[int]

    def __init__(self, num_nodes: int, num_inputs: int, activation_function: ActivationFunctionType):
        self.weights = np.random.uniform(-1, 1, (num_inputs, num_nodes)).astype(np.float16)
        self.biases = np.random.uniform(-1, 1, (1, num_nodes)).astype(np.float16)
        self.activation_function = activation_function
        self.shape = self.weights.shape
        self.inputs = num_inputs
    
    def pass_input(self, input_array: npt.NDArray[np.float16]) -> np.ndarray:
        z = np.matmul(input_array, self.weights)
        z = np.add(z, self.biases)
        z = self.activation_function(z)
        return z
    
    def __str__(self):
        return f"Shape: {self.shape}"

class Network_1D:
    layers: list[FCL_1D]

    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_layer: npt.NDArray[np.float16]) -> npt.NDArray[np.float16]:
        output = input_layer
        for layer in self.layers:
            output = layer.pass_input(output)
        return output
    
    def train(self, training_data: ClassificationTrainingData, epochs: int, train_test_split_ratio: float = 0.5, learning_rate: float = 1e-4) -> None:
        train_size = int(len(training_data.data) * train_test_split_ratio)
        random.shuffle(training_data.data)
        train_data = training_data.data[:train_size]
        test_data = training_data.data[train_size:]
        for epoch in range(epochs):
            total_loss = 0.0
            for datapoint in train_data:
                # forward pass
                activations = [datapoint.independents]
                zs = []
                for layer in self.layers:
                    z = np.matmul(activations[-1], layer.weights) + layer.biases
                    a = layer.activation_function(z)
                    zs.append(z)
                    activations.append(a)
                y_hat = activations[-1]
                y = datapoint.expected_out_array(len(training_data._classes))
                # loss MSE
                loss = 0.5 * np.sum((y - y_hat)**2)
                total_loss += loss
                # backward pass
                # start with output layer
                dL_dz = (y_hat - y) * y_hat * (1 - y_hat)  # for sigmoid
                # update output layer
                layer = self.layers[-1]
                dL_dW = np.outer(activations[-2], dL_dz)
                dL_dB = dL_dz.copy()
                layer.weights -= learning_rate * dL_dW
                layer.biases -= learning_rate * dL_dB
                # now hidden layers
                for i in reversed(range(len(self.layers) - 1)):
                    layer = self.layers[i]
                    next_layer = self.layers[i+1]
                    dL_da = np.matmul(dL_dz, next_layer.weights.T)
                    if layer.activation_function == ActivationFunctions.RELU:
                        da_dz = (zs[i] > 0).astype(np.float16)
                    else:
                        da_dz = np.ones_like(zs[i])
                    dL_dz = dL_da * da_dz
                    dL_dW = np.outer(activations[i], dL_dz)
                    dL_dB = dL_dz.copy()
                    layer.weights -= learning_rate * dL_dW
                    layer.biases -= learning_rate * dL_dB
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

if __name__ == "__main__":
    input_data = ClassificationTrainingData('Class_Seeds.csv')
    expected_output_4 = input_data.data[4].expected_out_array(len(input_data._classes))
    input_layer_4 = input_data.data[4].independents
    expected_output_5 = input_data.data[5].expected_out_array(len(input_data._classes))
    input_layer_5 = input_data.data[5].independents
    fclayer_1 = FCL_1D(
        num_nodes=10,
        num_inputs=7,
        activation_function=ActivationFunctions.RELU
    )
    fclayer_2 = FCL_1D(
        num_nodes=10,
        num_inputs=10,
        activation_function=ActivationFunctions.RELU
    )
    output_layer = FCL_1D(
        num_nodes=len(input_data._classes),
        num_inputs=10,
        activation_function=ActivationFunctions.SIGMOID
    )
    network = Network_1D(
        layers=[fclayer_1, fclayer_2, output_layer]
    )

    prediction = network.predict(input_layer_4)
    print(f"Output 4: {prediction.tolist()}")
    print(f"Expected output 4: {expected_output_4.tolist()}")

    network.train(input_data, 1500)
    prediction = network.predict(input_layer_4)
    print(f"Output 4: {prediction.tolist()}")
    print(f"Expected output 4: {expected_output_4.tolist()}")
    prediction = network.predict(input_layer_5)
    print(f"Output 5: {prediction.tolist()}")
    print(f"Expected output 5: {expected_output_5.tolist()}")