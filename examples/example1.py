from core.activations.relu import ReLU
from core.activations.softmax import Softmax
from core.layers.dense import Dense
from utils.datasets import create_data_spiral
from utils.plot import plot_data


if __name__ == "__main__":
    num_samples = 100
    num_classes = 3
    X, y = create_data_spiral(num_samples, num_classes)
    # plot_data(X, y)

    layer1 = Dense(2, 5)
    activation1 = ReLU()
    layer2 = Dense(5, 3)
    activation2 = Softmax()

    out = layer1.forward(X)
    out = activation1.forward(out)
    out = layer2.forward(out)
    out = activation2.forward(out)

    print(out[:5])