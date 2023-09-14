import matplotlib.pyplot as plt

def plot_data(X, y):
    plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
    plt.show()