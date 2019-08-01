Installation Instructions:
1. Install conda
2. Import environment.yml into your conda environment.

File Summary:
1. gp_test: contains functions related to GPR for mapping

    a) noisy_samples(interest_var, noise_std, zero_threshold)
        - adds noise to lwc or vwind sample points

    b) RBF_kernel(sigma_f, sigma_f_bounds, lengthscale, lengthscale_bounds)
        - returns an RBF kernel with according to input lengthscales and sigma_f

    c) fit_gpr(coord, interest_data, mean_function, kernel, noise_std, n_restarts_optimizer)
        - fits GPR to data using kernel defined
        - takes into account dense prior if we have
        - put n_restarts_optimizer = 0 for no optimization

    d) predict_map(gpr, test_data, mean_function, grid_shape)
        - predicts map on test points with uncertainity values

    =>Test included in file

2. process_map: functions to obtain coarse cloud models from dense map

    a) com(coord, data)
        - Calculates Center of Mass!

    b) border_cs(data, cs_shape, cloud_extent, threshold=1e-5, c="Black")
        - Calculates Border of Cross Section from dense map

    c) confidence_border_cs(data, std_data, cs_shape, cloud_extent, std_factor=1, fig=True, threshold=1e-5, color="darkgray")
        - Calculates Confidence Border of Cross Section from dense map
        - Inputs:
           As defined in data_plus_uncertainity & border_cs
        - Outputs: 2 borders with 68% confidence (by default)

3. cloud_prior: Functions to make cloud prior models from data

    a) get_field_prior(base_ellipse_params, data, t, zStart)
        - makes a prior of how lwc or vwind varies with radius or a/b or fitted ellipse

    b) get_var_func_with_z(coords, data, t, zStart, depth)
        - makes prior of variation of interest variable like lwc with height

    c) fitEllipse(cont,method)
        - input cloud border coordinates
        - fits ellipse on cloud (or circle also if possible)

    d) get_ellipses_params(t, zStart, ySlice, xSlice, z_depth, data_shape, xyExtent)
        - Gets center, width, height and area array of ellipses of fitted ellipses

    e) estimate_LR_coef(x, y)
        - estimates linear regression coefficients by fitting line
        - returns slope(m) & intercept (c) of fitted line

    => Demo included in file

4. cloud_prior_test: Functions to use cloud priors models to predict dense prior

    a) predict_prior(dense_map, desired_z, sa_prior, var_z_prior, field_prior, t, zStart, coord_extent, data_extent, data_unit, data_shape)
        - predict dense prior map at desired height
        - main inputs are :
            * dense map (predicted via GP or from external sensors)
            * prior models: surface area prior, variation with height prior, field prior
            * time
            * start height
            * spatial extent
        - outputs a dictionary with coordinates as keys and predictions as values at height desired

    => Check Demo included in file


5. flight_moving_frame: contains way to move in a moving frame and gather data

    a) show_map(data, xy_extent, data_unit, data_extent, time_stamp, height)
        - shows a cloud map at desired spatial and temporal extents

    b) line_trajectory(xi, yi, t, theta=0, s=20)
        - makes drone follow a straight line
        - used for adding wind to trajectory

    c) circ_trajectory(xc, yc, t, r, v=1000)
        - makes circular trajectory with desired speed

    d) lemniscate_trajectory(xc, yc, t, r, v=1000)
        - move in lemniscate trajectory with desired speed and inputs

    => Check Demo included in file


