#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This code to create a custom MNIST dataset was made possible thanks to
 https://github.com/LaRiffle/collateral-learning . 
 
Important to know that aside the tampering I did on the build_dataset function
for my own application, I also had to change rgba_to_rgb. Indeed, the function
was working as desired on Jupyter but not on Spyder. Do not ask me why !
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import pickle
import random

from torch.utils.data import Dataset, DataLoader
import torch
import math
import os



"""PLOT FUNCTIONS TO VISUALIZE THE FONTS AND DATASETS"""
def show_original_font(family:str):
    """Plot the original numbers used to create the dataset"""
    
    plt.figure()
    plt.title(family)
    plt.text(0, 0.4, '1234567890', size=50, family=family)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"plots/{family}_original.png")
    
    
    
def show_dataset_samples(plot_name:str, data, target_char, family:str, 
    n_examples=20):
    """Create a figure with 20 subplots. Each for one dataset's sample."""

    n_rows = math.ceil(n_examples / 5)
    
    plt.figure(figsize=(14, 3*n_rows))
    plt.suptitle(family)
    
    for i_plot in range(n_examples):
        ax = plt.subplot(n_rows, 5, i_plot+1)
        image = 255 - data[i_plot]

        ax.set_title(f"{target_char[i_plot]}")
        ax.imshow(image, cmap='gist_gray')
        
    if plot_name != "": plt.savefig(f"plots/{plot_name}.png")
    
    
    
   
def convert_to_rgb(data):
    
    def rgba_to_rgb(rgba):
        return rgba[1:]

    return np.apply_along_axis(rgba_to_rgb, 2, data) 



def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)



def center(data):
    # Inverse black and white
    wb_data = np.ones(data.shape) * 255 - data
    
    # normalize
    prob_data = wb_data / np.sum(wb_data)
    
    # marginal distributions
    dx = np.sum(prob_data, (1, 2))
    dy = np.sum(prob_data, (0, 2))

    # expected values
    (X, Y, Z) = prob_data.shape
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))
    
    # Check bounds
    assert cx > X/4 and cx < 3 * X/4, f"ERROR: {cx} > {X/4} and {cx} < {3 * X/4}"
    assert cy > Y/4 and cy < 3 * Y/4, f"ERROR: {cy} > {Y/4} and {cy} < {3 * Y/4}"
    
    # print('Center', cx, cy)
    
    x_min = int(round(cx - X/4))
    x_max = int(round(cx + X/4))
    y_min = int(round(cy - Y/4))
    y_max = int(round(cy + Y/4))
    
    return data[x_min:x_max, y_min:y_max, :]
   


def create_transformed_digit(digit:int, size:float, rotation:float, family:str):
    
    fig = plt.figure(figsize=(2,2), dpi=28)
    fig.text(0.4, 0.4, str(digit), size=size, rotation=rotation, family=family)

    # Rm axes, draw and get the rgba shape of the digit
    plt.axis('off')
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Convert to rgb
    data = convert_to_rgb(data)

    # Center the data
    data = center(data)

    # Apply an elastic deformation
    data = elastic_transform(data, alpha=991, sigma=9)

    # Free memory space
    plt.close(fig)
    
    return data

    

def build_dataset(C:dict, plot_name="", std_size=2.5, 
    dataset_digits=[i for i in range(10)]):
    """build a dataset with `dataset_size` according to the chosen font
    and deformation. Only digits in `datasets_digits` are in the created 
    dataset."""
    
    numbers_str="".join([str(n) for n in C['numbers']])
    file_name=f"{C['font']}_{numbers_str}_{C['n_samples']}_{C['tilt']}"
    if plot_name=="": plot_name=f"{C['font']}_{numbers_str}_{C['tilt']}"
    
    if os.path.isfile('dataset/'+file_name+".pkl"):
        
        return 1
    
    
    
    if C['seed']: np.random.seed(C['seed'])
    
    #Make a plot of each original digit to know what they look like
    show_original_font(C['font'])
    
    list_X = []
    list_y= []
    
    for i in range(C['n_samples']):
        
        if i%10 == 0: print(round(i / C['n_samples'] * 100), '%')
        
        #Choosing a number at this step and its transformation characteristics
        rotation = C['tilt'] + np.random.normal(0, C['std_tilt'])
        size = 50 + np.random.normal(0, std_size) 
        digit = dataset_digits[np.random.randint(len(dataset_digits))]

        X=create_transformed_digit(digit, size, rotation, C['font'])

        # Append data to the datasets
        list_X.append(X[:,:,0])
        list_y.append([digit])
    
    #Make a plot of some samples in the output dataset    
    show_dataset_samples(plot_name, list_X, list_y, C['font'])
    
    #save the dataset
    array_X, array_y = np.array(list_X), np.array(list_y)
    save_dataset(file_name, array_X, array_y)
    
    return np.array(list_X), np.array(list_y)


#def build_dataset(dataset_size:int, family:str, std_rotation:float, 
#    plot_name:str, std_size=2.5, dataset_digits=[i for i in range(10)],
#    tilted_angle=0, seed=None):
#    """build a dataset with `dataset_size` according to the chosen font
#    and deformation. Only digits in `datasets_digits` are in the created 
#    dataset."""
#    if seed: np.random.seed(seed)
#    
#    #Make a plot of each original digit to know what they look like
#    show_original_font(family)
#    
#    #What we are looking for
#    dataset_data = []
#    dataset_target= []
#    
#    
#    for i in range(dataset_size):
#        
#        if i%10 == 0: print(round(i / dataset_size * 100), '%')
#        
#        #Choosing a number at this step and its transformation characteristics
#        rotation = tilted_angle + np.random.normal(0, std_rotation)
#        size = 50 + np.random.normal(0, std_size) 
#        digit = dataset_digits[np.random.randint(len(dataset_digits))]
#        letter = str(digit)
#
#        data=create_transformed_digit(letter, size, rotation, family)
#
#        # Append data to the datasets
#        dataset_data.append(data[:,:,0])
#        dataset_target.append([digit])
#    
#    print(dataset_target)
##    #Make a plot of some samples in the output dataset    
#    show_dataset_samples(plot_name,dataset_data,dataset_target,family)
#    
#    
#        
#    return np.array(dataset_data), np.array(dataset_target)



def save_dataset(dataset_name, feature_data,label_data):
    
    with open('dataset/'+dataset_name+'.pkl', 'wb') as output:
        dataset = feature_data, label_data
        pickle.dump(dataset, output)
        
        
       
class ClientForModifiedMnist(Dataset):
    """
    PyTorch needs DataLoaders as an input for the Networks
    This class is needed to transform the input arrays into an input for 
    the `DataLoader` object.
    """
    
    
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels
    
    
    def __len__(self):
        return len(self.features)

    
    def __getitem__(self,idx):
        
        #3D input 1x28x28
        sample_x=torch.Tensor([self.features[idx]])
        
        sample_y = torch.ones((1,), dtype=torch.long)
        sample_y=sample_y.new_tensor(self.labels[idx])
        
        return sample_x,sample_y
    
    

def train_test_sets(file_name,training_samples,normalized=False,batch_size=100,
        shuffle=True):
    """function that returns the training and testing set used for our NN.
    If `normalized`, then the dataset returned is normalized with mean 0 and 
    standard deviation 1."""
    
    print('dataset/'+file_name+'.pkl')
    
    with open('dataset/'+file_name+'.pkl', 'rb') as pickle_file:
        X,y = pickle.load(pickle_file)
        X_train,y_train=X[:training_samples],y[:training_samples]
        X_test,y_test=X[training_samples:],y[training_samples:]
                
     
    if normalized:
        
        #Obtain the mean and the standard deviation of each image feature
        mean_train=np.zeros((28,28))
        std_train=np.zeros((28,28))
       
        for i in range(28):
            for j in range(28):
                mean_train[i,j]=np.mean(X_train[:,i,j])
                std_train[i,j]=np.std(X_train[:,i,j])
        
        #Normalize the image        
        X_train_new=(X_train-mean_train)
        X_test_new=(X_test-mean_train)
        
        for i in range(28):
            for j in range(28):
                if std_train[i,j]!=0:
                    X_train_new[:,i,j]/=std_train[i,j]
                    X_test_new[:,i,j]/=std_train[i,j]
                
                elif std_train[i,j]==0:
                    X_test_new[:,i,j]=0
    
    elif not normalized:
        X_train_new=X_train/255
        X_test_new=X_test/255
        
    print(X_train_new.min(),X_train_new.max())
    print(X_test_new.min(),X_test_new.max())
            
    train=ClientForModifiedMnist(X_train_new,y_train)         
    train_dl=DataLoader(train,batch_size=batch_size,shuffle=shuffle)   
     
    test=ClientForModifiedMnist(X_test_new,y_test)         
    test_dl=DataLoader(test,batch_size=batch_size,shuffle=shuffle)  
    
    return train_dl,test_dl
          
        
        
        
        
def file_name(family,numbers_str,training_samples,tilted_angle=0):
    
    if tilted_angle==0:
        return family +" "+ numbers_str+" "+str(training_samples)
    else:
        return family +" "+ numbers_str+" "+str(training_samples)+"_"+str(tilted_angle)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

