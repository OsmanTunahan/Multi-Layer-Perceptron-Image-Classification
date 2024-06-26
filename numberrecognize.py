import math
from typing import List, Union

def relu(x: List[float]) -> List[float]:
    return [max(0, i) for i in x]

def sigmoid(x: List[float]) -> List[float]:
    return [0 if i <= -700 else (1 / (1 + math.exp(-i))) if -700 < i < 700 else 1 for i in x]

def multiply(a: float, b: float) -> float:
    return a * b

def linear(x: List[float], w: List[List[float]]) -> List[float]:
    return [sum(map(multiply, x, weights)) for weights in w]

def forward_pass(network: List[Union[List[Union[str, List[List[float]]]], str]], x_sample: List[float]) -> List[float]:
    x = x_sample
    for layer in network:
        if isinstance(layer, str):
            if "relu" in layer:
                x = relu(x)
            elif "sigmoid" in layer:
                x = sigmoid(x)
        elif isinstance(layer, list):
            for component in layer:
                if isinstance(component, str) and "linear" in component:
                    x = linear(x, layer[1])
    return x