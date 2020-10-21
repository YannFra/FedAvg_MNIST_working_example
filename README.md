# Federated Learning simple example

Simple example of federated learning on MNIST

FedAvg is the default optimization solver but FedProx can also be used with a $\mu$ different of 0.

All the clients are considered at every iteration. No clients sampling is available.

The considered model is a simple neural netwrok with 2 fully connected layers.

The algorithm by default save a couple of training parameters in `saved_exp_info`:
- the accuracy of the different participants at every iterations in `acc`. The saved object is a list of list where the first index is for the training iteration and the second index is for the considered client.
- the loss of the different participants at every iterations in `loss`.
- the local models of all the clients at every iteration in `local_model_history`.
- the global model obtained at the end of the training in `final_model`.
- the global models at every iteration in `server_history`.
