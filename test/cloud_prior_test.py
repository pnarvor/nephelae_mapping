#! /usr/bin/python3

import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import matplotlib
from netCDF4 import MFDataset
from nephelae_simulation.mesonh_interface import ScaledArray
from nephelae_simulation.mesonh_interface import DimensionHelper
from nephelae_simulation.mesonh_interface import MesoNHVariable
from nephelae_mapping.test.util import save_pickle, load_pickle
from nephelae_mapping.test.process_map import border_cs, com
from nephelae_mapping.test.flight_moving_frame import get_coordinate_extent, show_map
from nephelae_mapping.test.cloud_prior import fitEllipse, get_ellipse_coords
from nephelae_mapping.test.gp_test import get_2d_test_grid
from matplotlib.patches import Ellipse

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

if __name__ == "__main__":

    data_path = "exp/4/"
    save_path = "exp/4/"

    z_desired = 50 # 50m above cloud base

    # file names of prior in form of pickles
    sa_prior_name = "SA_prior_params"      # surface area prior with z
    field_prior_name = "lwc_field_prior"   # Field prior at a cross section
    var_z_prior_name = "var_z_prior"       # Variation of interest variable at center with Z

    sa_prior    = load_pickle(data_path + sa_prior_name)
    field_prior = load_pickle(data_path + field_prior_name)
    var_z_prior = load_pickle(data_path + var_z_prior_name)

    atm = MFDataset("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc")
    tvar, zvar, xvar, yvar = get_coordinate_extent(atm)

    var_interest = 'RCT' # or 'WT' etc
    data = MesoNHVariable(atm, var_interest, interpolation='linear')
    lwc_unit = "kg a/kg w"

    t = 211        # inital timestamp
    zStart = 1002  # initial height  ini m
    z_depth = 100  # length of z to travel in cloud
    xSlice = slice(1702, 2172, None)
    ySlice = slice(4170, 4700, None)
    lwc_extent = [data[t,:,xSlice,ySlice].data.min(), data[t,:,xSlice,ySlice].data.max() + 0.25 * data[t,:,xSlice,ySlice].data.max()]

    xyBounds = data[t, zStart, xSlice, ySlice].bounds
    xyExtent = [xyBounds[0][0], xyBounds[0][1], xyBounds[1][0], xyBounds[1][1]]
    xExtent = xvar[(xvar >= xyExtent[0]) & (xvar <= xyExtent[1])]
    yExtent = yvar[(yvar >= xyExtent[2]) & (yvar <= xyExtent[3])]
    cloud_2D_coords, coord_grid = get_2d_test_grid(xExtent, yExtent)
    data_shape = data[t, zStart, ySlice, xSlice].data.shape

    input_cloud_base_data = data[t, zStart, ySlice, xSlice].data
    fig, lwc_data = show_map(input_cloud_base_data.T, xyExtent, lwc_unit, lwc_extent, t, zStart)
    _, border_data = border_cs(input_cloud_base_data, data_shape, xyExtent, threshold=3e-5, c="Black")
    curve_coords = border_data._get_allsegs_and_allkinds()[0][0][0]  # only valid when only 1 curve inside a surface

    # Fit Ellipse on curve coords
    ell_params, ellipse, fit_score = fitEllipse(curve_coords, 1)
    plt.plot(ellipse[:,0], ellipse[:,1])
    plt.scatter(ell_params[0], ell_params[1], c="Red")

    base_surface_area = np.pi * ell_params[2] * ell_params[3]

    # lwc or vwind value of center of cloud base
    cloud_base_center_data = data[t, zStart, ell_params[1], ell_params[0]]

    #===========Predict cloud surface area at desired z===================
    sa_per_at_z = sa_prior[0] * z_desired + sa_prior[1]
    sa_at_z = sa_per_at_z / 100 * base_surface_area

    # Fetch cloud ellipse shape coordinates
    ell_a =  sa_per_at_z / 100 * ell_params[2]
    ell_b = sa_per_at_z / 100 * ell_params[3]
    new_ellipse_params = [ell_params[0], ell_params[1], ell_a, ell_b, ell_params[4]]

    predicted_shape = get_ellipse_coords(new_ellipse_params)
    plt.figure()
    plt.plot(predicted_shape[:,0], predicted_shape[:,1])
    plt.scatter(new_ellipse_params[0], new_ellipse_params[1], c="white") #center
    plt.xlim(xyExtent[0], xyExtent[1])
    plt.ylim(xyExtent[2], xyExtent[3])

    #===============Predict cloud center dense point=====================
    ccenter_var_pred = var_z_prior[z_desired] * cloud_base_center_data

    #===============Use Field Prior to get LWC or VWIND Maps========

    a_by_b = ell_a / ell_b # ellipse radii ratio

    pred_value_dict = {}

    # Initilaize all 2 points with 0 value
    for x, y in cloud_2D_coords:
        pred_value_dict[x, y] = 0

    for i in range(1, int(ell_b) + 1):

        # Fetch ellipse params for each new ellipse
        pred_el_params = [new_ellipse_params[0], new_ellipse_params[1], a_by_b * i, i, new_ellipse_params[4]]

        # Get ellipse coords
        ell_coords = get_ellipse_coords(pred_el_params)

        # Get fraction value of b compared to cloud base ellipse
        pred_el_frac = round(i / int(ell_b), 2)

        # Find key near to our fraction value
        field_key = field_prior[min(field_prior.keys(), key=lambda k: abs(k - pred_el_frac))]

        # Get estimated value of lwc or vwind for points in this ellipse
        pred_for_ellipse = field_prior.get(pred_el_frac, field_key) * cloud_base_center_data

        # Find nearest coordinates of ellipse points in 2D grid & assign predicted values
        for x, y in ell_coords:
            pred_value_dict[find_nearest(xExtent, x), find_nearest(yExtent, y)] = pred_for_ellipse

    # Reshape in grid form
    dense_prior = np.array(list(pred_value_dict.values())).reshape(coord_grid.shape[:2])

    # show dense prior
    show_map(dense_prior, xyExtent, lwc_unit, lwc_extent, t, zStart + z_desired, new_fig=False)