# Federated Learning simple implementation

Working implementation of FedAvg (mu=0) and FedProx using PyTorch. FedAvg is the default optimizer solver. No sampling is available and all the clients are considered at every iteration.

Two examples on which to build are considered:
- `FL_MNIST.py`: fully connected 2 layers neural network tested on MNIST.
- `FL_MNIST_custom.py`: neural network tested on what we call custom MNIST.

The computed federated learning saves by default a couple of training parameters in `saved_exp_info`:
- the accuracy of the different participants at every iterations in `acc`. The saved object is a list of list where the first index is for the training iteration and the second index is for the considered client.
- the loss of the different participants at every iterations in `loss`.
- the local models of all the clients at every iteration in `local_model_history`.
- the global model obtained at the end of the training in `final_model`.
- the global models at every iteration in `server_history`.

Custom MNIST enables creating for each client a MNIST dataset where each client can have a different font and/or a rotation angle. We propose in `FL_MNIST_custom.py` a simple example for FedAvg with two clients having same digits but rotated by 30 degrees for one client. More info about how the font impacts the obtained dataset can be found [here](https://github.com/LaRiffle/collateral-learning/blob/a8e40193e234e331fe49a5b0e1207b34464efa16/tutorials/Part%2001%20-%20Building%20a%202%20target%20features%20dataset.ipynb#L77).

A lot of fonts are available. We recommend 'InconsolataN' and 'jsMath-cmti10'. 'InconsolataN' have continuous 4 and bars on th 0 while 'jsMath-cmti10' have discontinuous 4s and no bar on the 0.
