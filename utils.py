import pickle as pkl
from typing import Dict, List, Union, Callable

def load_dataset(filename: str = "mnist.pkl") -> Dict[str, Union[List, int]]:
    """Load MNIST images & labels from a pickle file.

       Input:
        - filename: A string for the name of the pickle file containing the MNIST samples.

       Output:
        - data_dict: A dictionary that contains the images and the labels.
    """
    with open(filename, 'rb') as infile:
        data_dict = pkl.load(infile)
    return data_dict

def load_network(filename: str = "network_3layer.pkl") -> List[Union[List, str]]:
    """Load a network from a pickle file.

       Input:
        - filename: A string for the name of the pickle file containing the network.

       Output:
        - network: A list of layers, e.g. 
                    [['linear', Weights], 'relu', ['linear', Weights], ...]
    """
    with open(filename, 'rb') as infile:
        network = pkl.load(infile)
    return network

def display_image(X: List[int]) -> None:
    """Display an image using ASCII chars.
       
       Input:
        - X: An image as a vector; i.e. a list with D elements.

       Output: None.
    """
    for i in range(0, 28 * 28):
        if i % 28 == 0 and i > 0: 
            print("")
        print("." if X[i] < 125 else "@", end="")
    print("")

def display_network(network: List[Union[List, str]]) -> None:
    """Display a network's layers and layer sizes.
       
       Input:
        - network: A list of layers, e.g. 
                    [['linear', Weights], 'relu', ['linear', Weights], ...]

       Output: None.
    """
    layer_info = [
        layer[0] + ": " + str(len(layer[1][0])) + "->" + str(len(layer[1])) if isinstance(layer, list) else layer
        for layer in network
    ]
    print(layer_info)

def calculate_accuracy(dataset: Dict[str, Union[List, int]], network: List[Union[List, str]], predictor: Callable[[List[Union[List, str]], List[int]], List[float]]) -> float:
    """Calculate the accuracy of a network's predictions.
       
       Input:
        - dataset: A dictionary that contains the images and the labels.
        - network: A list of layers, e.g. 
                    [['linear', Weights], 'relu', ['linear', Weights], ...]
        - predictor: The forward-pass function that takes the network and a sample, and returns the outputs
                    of the last layer.
       Output: Accuracy (%).
    """
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    N = len(X_test)

    correct = 0
    for i in range(N):
        X = X_test[i]
        y = y_test[i]
        output = predictor(network, X)
        y_pred = output.index(max(output))
        if y == y_pred: 
            correct += 1

    return (correct / N) * 100