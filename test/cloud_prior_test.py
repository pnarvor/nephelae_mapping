#! /usr/bin/python3

import os
import time
import numpy as np
import math as m
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
from nephelae_mapping.test.flight_moving_frame import get_coordinate_extent, show_map, lemniscate_trajectory, rotate
from nephelae_mapping.test.cloud_prior import fitEllipse, get_ellipse_coords
from nephelae_mapping.test.gp_test import get_2d_test_grid, noisy_samples, RBF_kernel, fit_gpr, predict_map
from matplotlib.patches import Ellipse

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or m.fabs(value - array[idx-1]) < m.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def get_sampling_data(data, t, z, initial_pos, num_points, radius, v_circle, show_fig=True):

    XT = np.empty((0, 2), float)
    yT = []
    xi = initial_pos[0]
    yi = initial_pos[1]
    rot_angle = 0
    sub_factor = 0
    xf = 0
    yf = 0
    rot_active = False
    angles = [90, 45, 135]
    colors = ["red", "green", "blue", "yellow", "black"]
    color = "white"

    for i in range(1, num_points+1):
        xn, yn = lemniscate_trajectory(xi, yi, i - sub_factor, radius, v_circle)
        if (xn == xf and yn == yf):
            rot_active = True
            sub_factor = i - 1
            if (angles):
                rot_angle = angles.pop(0)
                color = colors.pop(0)
            else:
                rot_active = False
                angles = [90, 45, 135]
                rot_angle = 0
                color = "white"
                colors = ["red", "green", "blue", "yellow", "black"]
        if (i == 1):
            xf, yf = xn, yn
        if (rot_active):
            xn, yn = rotate((xi, yi), (xn, yn), m.radians(rot_angle))

        XT = np.vstack([XT, [xn, yn]])
        yT = np.append(yT, data[t, z, yn, xn])

        if(show_fig):
            plt.scatter(xn, yn, c=color, marker="x", s=15)

    return XT, yT.reshape(-1, 1)

def predict_prior(dense_map, desired_z, sa_prior, var_z_prior, field_prior, t,
                  zStart, coord_extent, data_extent, data_unit, data_shape, thrs=10e-5):

    fig, lwc_data = show_map(dense_map.T, coord_extent, data_unit, data_extent, t, zStart)
    _, border_data = border_cs(dense_map, data_shape, coord_extent, threshold=thrs, c="Black")
    curve_coords = border_data._get_allsegs_and_allkinds()[0][0][0]  # only valid when only 1 curve inside a surface

    # Fit Ellipse on curve coords
    ell_params, ellipse, fit_score = fitEllipse(curve_coords, 1)
    plt.plot(ellipse[:,0], ellipse[:,1])
    plt.scatter(ell_params[0], ell_params[1], c="Red")

    base_surface_area = np.pi * ell_params[2] * ell_params[3]

    # lwc or vwind value of center of cloud base
    cloud_base_center_data = lwc_data.get_array()[int((ell_params[1] - coord_extent[2]) / 25), int((ell_params[0] - coord_extent[0]) / 25)]
    #cloud_base_center_data = data[t, zStart, ell_params[1], ell_params[0]]

    #===========Predict cloud surface area at desired z===================
    sa_per_at_z = sa_prior[0] * desired_z + sa_prior[1]
    sa_at_z = sa_per_at_z / 100 * base_surface_area

    # Fetch cloud ellipse shape coordinates
    ell_a =  sa_per_at_z / 100 * ell_params[2]
    ell_b = sa_per_at_z / 100 * ell_params[3]
    new_ellipse_params = [ell_params[0], ell_params[1], ell_a, ell_b, ell_params[4]]

    predicted_shape = get_ellipse_coords(new_ellipse_params)
    plt.figure()
    plt.plot(predicted_shape[:,0], predicted_shape[:,1])
    plt.scatter(new_ellipse_params[0], new_ellipse_params[1], c="white") #center
    plt.xlim(coord_extent[0], coord_extent[1])
    plt.ylim(coord_extent[2], coord_extent[3])

    #===============Predict cloud center dense point=====================
    ccenter_var_pred = var_z_prior[desired_z] * cloud_base_center_data

    #===============Use Field Prior to get LWC or VWIND Maps========

    a_by_b = ell_a / ell_b # ellipse radii ratio

    pred_value_dict = {}

    # Initilaize all 2D points with 0 value
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
        pred_for_ellipse = field_prior.get(pred_el_frac, field_key) * ccenter_var_pred

        # Find nearest coordinates of ellipse points in 2D grid & assign predicted values
        for x, y in ell_coords:
            pred_value_dict[find_nearest(xExtent, x), find_nearest(yExtent, y)] = pred_for_ellipse

    # Reshape in grid form
    dense_prior = np.array(list(pred_value_dict.values())).reshape(coord_grid.shape[:2])

    # show dense prior
    show_map(dense_prior, coord_extent, data_unit, data_extent, t, zStart + desired_z, new_fig=False)

    if(save_plots):
        plt.savefig(save_path + "dense_gp_prior.png")

    if(dense_prior_save):
        save_pickle(pred_value_dict, save_path + "dense_prior_dict")

    return pred_value_dict


if __name__ == "__main__":

    data_path = "exp/4/"
    save_path = "exp/4/"
    dense_prior_save = False
    save_plots = False
    z_desired = 50 # m above cloud base

    ##================Load Models or Functions======================
    ## file names of prior in form of pickles
    sa_prior_name = "SA_prior_params"      # surface area prior with z
    field_prior_name = "lwc_field_prior"   # Field prior at a cross section
    var_z_prior_name = "var_z_prior"       # Variation of interest variable at center with Z

    sa_prior    = load_pickle(data_path + sa_prior_name)
    field_prior = load_pickle(data_path + field_prior_name)
    var_z_prior = load_pickle(data_path + var_z_prior_name)
    ##==============================================================

    ##==========Load MesoNH data and coords=========================
    atm = MFDataset("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc")
    tvar, zvar, xvar, yvar = get_coordinate_extent(atm)

    var_interest = 'RCT' # or 'WT' etc
    data = MesoNHVariable(atm, var_interest, interpolation='linear')
    lwc_unit = "kg a/kg w"

    ##======== Define t, z , x and y values to work on================
    t = 211                            # inital timestamp
    zStart = 1002                      # initial height  ini m
    z_depth = 100                      # length of z to travel in cloud
    xSlice = slice(1702, 2172, None)   # x range in m
    ySlice = slice(4170, 4700, None)   # y range in m
    ##================================================================

    ##========Get Grid Coordinates and extents of data================
    lwc_extent = [data[t,:,xSlice,ySlice].data.min(), data[t,:,xSlice,ySlice].data.max() + 0.25 * data[t,:,xSlice,ySlice].data.max()]
    xyBounds = data[t, zStart, xSlice, ySlice].bounds
    xyExtent = [xyBounds[0][0], xyBounds[0][1], xyBounds[1][0], xyBounds[1][1]]
    xExtent = xvar[(xvar >= xyExtent[0]) & (xvar <= xyExtent[1])]
    yExtent = yvar[(yvar >= xyExtent[2]) & (yvar <= xyExtent[3])]
    cloud_2D_coords, coord_grid = get_2d_test_grid(xExtent, yExtent)
    data_shape = data[t, zStart, ySlice, xSlice].data.shape
    ##================================================================

    ##============= Input Dense Map ==================================

    ##================================================================
    ## Case 1 : Without using dense map (use some external measurements + models)
    ## Here I simply use the GT to compare if dense_gp_prior matches GT
    #input_cloud_base_data = data[t, zStart, ySlice, xSlice].data
    #predict_prior(input_cloud_base_data, z_desired, sa_prior, var_z_prior, field_prior, t,
    #               zStart, xyExtent, lwc_extent, lwc_unit, data_shape)
    ##=================================================================

    ##================================================================
    ## Case 2 : Do GP maaping at certain height and use models on it to get dense prior at different height

    input_cloud_base_data = data[t, zStart, ySlice, xSlice].data
    show_map(input_cloud_base_data.T, xyExtent, lwc_unit, lwc_extent, t, zStart)

    # Sample data using dummy leminisacte trajectory
    sample_coord, measured_values = get_sampling_data(data, t, zStart, [1925, 4400], 300, 140, 800)

    # Define GP params
    noise_std = 1e-6
    zero_thresholding = True
    sigma_f = 1e-4
    sigma_f_bounds = (1e-2, 9)
    lengthscale = 50
    lengthscale_bounds = (5, 500)
    dense_prior_for_gpr ={}

    # Add noise to data
    sample_with_noise = noisy_samples(measured_values, noise_std, zero_thresholding)

    kernel = RBF_kernel(sigma_f, sigma_f_bounds, lengthscale, lengthscale_bounds)

    gpr = fit_gpr(sample_coord, measured_values, dense_prior_for_gpr, kernel, noise_std, 0)

    # Predict Map and show
    pred, pred_grid, std_pred = predict_map(gpr, cloud_2D_coords, dense_prior_for_gpr, coord_grid.shape[:2])
    show_map(pred_grid, xyExtent, lwc_unit, lwc_extent, t, zStart)
    std_pred = std_pred.reshape(pred_grid.shape)
    show_map(std_pred, xyExtent, lwc_unit, lwc_extent, t, zStart)

    # Predict Prior at desired height through this dense map
    predict_prior(pred_grid.T, z_desired, sa_prior, var_z_prior, field_prior, t,
                  zStart, xyExtent, lwc_extent, lwc_unit, data_shape)
