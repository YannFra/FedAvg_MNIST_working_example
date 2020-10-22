# Federated Learning simple example

Simple example of federated learning on MNIST using PyTorch.

FedAvg is the default optimization solver but FedProx can also be used with a $\mu$ different of 0.

All the clients are considered at every iteration. No clients sampling is available.

The considered model is a simple neural netwrok with 2 fully connected layers.

The algorithm by default save a couple of training parameters in `saved_exp_info`:
- the accuracy of the different participants at every iterations in `acc`. The saved object is a list of list where the first index is for the training iteration and the second index is for the considered client.
- the loss of the different participants at every iterations in `loss`.
- the local models of all the clients at every iteration in `local_model_history`.
- the global model obtained at the end of the training in `final_model`.
- the global models at every iteration in `server_history`.


We propose in `FL_MNIST_custom.py` a simple example for FedAvg with two clients having same digits but rotated by 30 degrees for one client. We use a framework to create specific client dataset where the differences can be controlled. Each clients' dataset can be differenciated with two indicators: `font` and `rotation`. For `font` more details can be found [here](https://github.com/LaRiffle/collateral-learning/blob/a8e40193e234e331fe49a5b0e1207b34464efa16/tutorials/Part%2001%20-%20Building%20a%202%20target%20features%20dataset.ipynb#L77) while `rotation` rotates each dataset images with the chosen rotation angle. 

A lot of fonts are avaailable. We recommend 'InconsolataN' and 'jsMath-cmti10'. 'InconsolataN' have continuous 4 and bars on th 0 while 'jsMath-cmti10' have discontinuous 4s and no bar on the 0.
