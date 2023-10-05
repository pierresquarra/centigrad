# centigrad: Micrograd, just a little bigger

After watching Andrej Karpathy's excellent [micrograd YouTube video](https://youtu.be/VMj-3S1tku0?si=1Ie6yDJPs_dbI04f), I decided to give it a shot myself. The autograd engine is basically copied and extended with some more functions. The way I implemented the building blocks for the neural nets, resemble PyTorches API. This "deep learning framework" is hell of a lot slower than PyTorch, but it was a fun project to do.

## Features

- Computational graph visualization, kind of like Andrej did it in [micrograd](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)
- Histogram plotting, just like Andrej did it in [makemore](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore)
- MNIST example (very very slow)
- makemore example (very very very slow)
