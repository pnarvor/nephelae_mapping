################################################################
# Calculates macroscopic cloud properties from dense cloud map #
################################################################

import numpy as np
import matplotlib.pyplot as plt

def com(coord, data):
    # Calculates Center of Mass!
    # Inputs:
    #   data - numpy list of data of interest : shape (n,), 1d array
    #   coord - numpy list of points(Point=(x,y,z,..)) : shape (n,D) Dd array
    mass_sum = np.sum(data)
    ndim = coord.shape[1]
    return [np.sum(data * coord[:,i])/ mass_sum for i in range(ndim)]

def border_cs(data, cs_shape, cloud_extent, threshold=1e-5, c="Black"):
    # Calculates Border of Cross Section from dense map
    # Inputs:
    #   data - numpy list of data of interest: shape (n,1) 1 column matrix
    #   cs_shape - cross_section shape (eg: 70*70 square)
    #   cloud_extent - [min(x), max(x), min(y), max(y)] in kms for plot
    #   fig - if figure is needed
    #   threshold - value to threshold
    # Outputs: border grid plus optional plot

    data[data < threshold] = 0  # Thresholding
    data_grid = np.reshape(data, cs_shape)  # reshaping as a cs grid
    border_plot = plt.contour(data_grid, origin='lower', extent=cloud_extent, colors=c, levels=0)
    plt.xlim(cloud_extent[0], cloud_extent[1])
    plt.ylim(cloud_extent[2], cloud_extent[3])
    return data_grid, border_plot

def data_plus_uncertainity(data, std_data, std_factor=1):
    # Data +/- sigma (or 2 sigma, 3 sigma,..)
    # Inputs:
    #   data - numpy list of data of interest : shape (n,1) 1 column matrix
    #   std_data - std deviation list of same shape
    #   std_factor - 1,2,3,..(1 sigma, 2 sigma,..)
    # Outputs: data - uncertainty, data + uncertainty

    return data - std_factor * std_data, data + std_factor * std_data

def confidence_border_cs(data, std_data, cs_shape, cloud_extent, std_factor=1, fig=True, threshold=1e-5, color="darkgray"):
    # Calculates Confidence Border of Cross Section from dense map
    # Inputs:
    #   As defined in data_plus_uncertainity & border_cs
    # Outputs: 2 borders with 68% confidence (by default)

    inner_border_data, outer_border_data = data_plus_uncertainity(data, std_data, std_factor)
    border_cs(inner_border_data, cs_shape, cloud_extent, fig, threshold, color)
    border_cs(outer_border_data, cs_shape, cloud_extent, fig, threshold, color)