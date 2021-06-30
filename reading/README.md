
# Reading List and Comments

- [Article for FastTensorLoader](https://towardsdatascience.com/better-data-loading-20x-pytorch-speed-up-for-tabular-data-e264b9e34352)
This article gives a good alternative to the pytorch in-built dataLoader; however, it's drawbacks are that we need to load the dataset at once (not feasible for our purposes) and it does not have multithreading support like pytorch does. This is a good starting point for a hybrid dataLoader that combines the benefits of both.

- [Using GANs for numerical integration](https://arxiv.org/pdf/1707.00028.pdf)
The methodology is very interesting where they use GANs to approximate the distribution of the function being integrated and then use MC integration. Still unclear whether this method is feasible for MEM calculations.
