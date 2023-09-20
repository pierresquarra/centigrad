class _Optimizer:
    """A private base class for optimization algorithms."""

    def __init__(self, params, lr=0.1, momentum=0.0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum

    def step(self) -> None:
        """Updates the parameters using the specified optimization algorithm."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Resets the gradients of the parameters to zero."""
        for p in self.params:
            p.grad = 0.0


class SGD(_Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer."""

    def __init__(self, params, lr=0.1, momentum=0.0) -> None:
        super().__init__(params, lr, momentum)

    def step(self) -> None:
        """Updates the parameters using the SGD optimization algorithm."""
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """Resets the gradients of the parameters to zero."""
        return super().zero_grad()
