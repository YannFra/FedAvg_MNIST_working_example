#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_samples(dataset, title=None, plot_name="", n_examples=20):
    
    n_rows = abs(n_examples / 5)
    plt.figure(figsize=(3* n_rows, 3*n_rows))
    if title: plt.suptitle(title)
    
    for idx,(X,y) in enumerate(dataset.dataset):
        
        if idx<n_examples:
    
            ax = plt.subplot(n_rows, 5, idx + 1)
            ax.set_title(f"{y}")
            
            image = 255 - X.view((28,28))
            ax.imshow(image, cmap='gist_gray')

    if plot_name!="":plt.savefig(f"plots/"+plot_name+".png")
