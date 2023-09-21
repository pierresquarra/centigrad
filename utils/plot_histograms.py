import matplotlib.pyplot as plt
import numpy as np


def activation_distribution(layers, layer_type):
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]):  # note: exclude the output layer
        if isinstance(layer, layer_type):
            t = np.vectorize(lambda x: x.data)(layer.out)
            print('layer %d (%s): mean %+.2f, std %.2f, saturated: %.2f%%' % (
            i + 1, layer.__class__.__name__, t.mean(), t.std(), (np.abs(t) > 0.97).astype(np.float32).mean() * 100))
            hy, hx = np.histogram(t, density=True)
            plt.plot(hx[:-1], hy)
            legends.append(f'layer {i + 1} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('activation distribution')


def gradient_distribution(layers: list, layer_type):
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]):  # note: exclude the output layer
        if isinstance(layer, layer_type):
            t = np.vectorize(lambda x: x.grad)(layer.out)
            print('layer %d (%s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = np.histogram(t, density=True)
            plt.plot(hx[:-1], hy)
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends);
    plt.title('gradient distribution')
