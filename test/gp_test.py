from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import random
from scipy import ndimage
import pickle
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import MFDataset

def load_window_size(dataset_path, x_indices, y_indices):

    atm = MFDataset(dataset_path)
    xvar = 1000.0 * atm.variables['W_E_direction'][:]
    yvar = 1000.0 * atm.variables['S_N_direction'][:]

    # As in GT window
    x_extent = xvar[x_indices[0] : x_indices[1]]
    y_extent = yvar[y_indices[0] : y_indices[1]]
    return x_extent, y_extent

def load_samples(coord_pickle, var_pickle):
    #load coordinates + var (LWC, VWind) from pickle##

    pickle_in = open(coord_pickle, "rb")
    coordinates = pickle.load(pickle_in) #, encoding='latin1')
    pickle_in.close()

    pickle_in = open(var_pickle, "rb")
    interest_var = pickle.load(pickle_in) #, encoding='latin1')
    pickle_in.close()

    return coordinates, interest_var

def plot_sampled_points(coord, interest_var, x_extent, y_extent, data_extent, data_unit, save_fig = False):

    plt.figure(figsize = (12,4))
    plt.title("Acquired GT Data : %d points with 1 point/sec rate, t=(%ds, %ds)"%(interest_var.shape[0], coord[0,2], coord[-1,2]))
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.xlim(x_extent[0], x_extent[-1])
    plt.ylim(y_extent[0], y_extent[-1])
    plt.scatter(coord[:,0], coord[:,1], marker="x", c=interest_var[:,0], s=15, vmin = data_extent[0], vmax = data_extent[1], cmap=plt.cm.viridis)
    cbar = plt.colorbar(fraction=0.0471, pad=0.01)
    cbar.ax.set_title(data_unit, pad=14)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.tight_layout()
    if(save_fig):
        plt.savefig("GT_points")

def noisy_samples(interest_var, noise_std, zero_threshold):

    interest_var = interest_var + (noise_std ** 2 * np.random.randn(len(interest_var))).reshape(-1,1) #added noise
    if(zero_threshold): # threshold negative values to 0
        interest_var = interest_var.clip(min=0) #ignoring negative rct values
    return interest_var

def RBF_kernel(sigma_f, sigma_f_bounds, lengthscale, lengthscale_bounds):

    return C(constant_value = sigma_f ** 2, constant_value_bounds = (sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2) ) * \
            RBF(length_scale = lengthscale, length_scale_bounds = lengthscale_bounds )

def fit_gpr(coord, interest_data, kernel, noise_std, n_restarts_optimizer):

    # Define and Fit GPR
    if(n_restarts_optimizer==0):
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, n_restarts_optimizer=0, optimizer=None).fit(coord, interest_data)
    else:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_std ** 2, n_restarts_optimizer=n_restarts_optimizer).fit(coord, interest_data)
    return gpr

def get_2d_test_grid(x_extent, y_extent):
    XY_grid = np.array(np.meshgrid(x_extent, y_extent)).T
    XY_test = XY_grid.reshape(-1,2)
    return XY_test, XY_grid

def get_3d_test_data(test_2d_coords, time_stamp):

    # Add a time column associated to a 2D data
    # time_stamp is any time to append in secoonds
    # Eg: time_stamp=394 will append a column of 394 to 2D(x,y) data

    XYT_test = np.concatenate((test_2d_coords , np.full((test_2d_coords.shape[0], 1),time_stamp)),axis=1)
    return XYT_test

def predict_map(gpr, test_data, grid_shape):

    pred, std_pred = gpr.predict(test_data, return_std=True)
    pred_grid = np.reshape(pred, grid_shape)
    return pred, pred_grid, std_pred

def show_map(pred_grid, xy_extent, data_unit, data_extent, time_stamp):

    fig = plt.figure(figsize=(12,4))
    plt.title("Prediction at t = %ds" % (time_stamp))
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.imshow(pred_grid.T, origin='lower', extent=xy_extent, cmap=plt.cm.viridis)

    cbar = plt.colorbar(fraction=0.0471, pad=0.01)
    cbar.ax.set_title(data_unit, pad=14)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.clim(data_extent[0], data_extent[1])
    plt.tight_layout()

if __name__ == "__main__":

    coord, lwc_data = load_samples("coord.pickle", "lwc_data.pickle")

    lwc_extent = [0, 4.5e-4] # range of lwc
    lwc_unit = "kg a/kg w"
    noise_std = 1e-6
    zero_thresholding = True

    x_extent, y_extent = load_window_size("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc", [60,150], [164,186])

    # Plot GT points
    plot_sampled_points(coord, lwc_data, x_extent, y_extent, lwc_extent, lwc_unit, save_fig = False)

    # Add noise to data
    lwc_data =  noisy_samples(lwc_data, noise_std, zero_thresholding)

    # GPR Params
    sigma_f = 1e-4
    sigma_f_bounds = (1e-2, 9)
    lengthscale = [50, 50, 20]
    lengthscale_bounds = [(5, 500), (5, 500), (1, 300)]

    kernel = RBF_kernel(sigma_f, sigma_f_bounds, lengthscale, lengthscale_bounds)
    gpr = fit_gpr(coord, lwc_data, kernel, noise_std, 0)

    # Test Data
    test_time_stamp = coord[-1, 2] # second of last sample
    test_2D_coords, test_grid = get_2d_test_grid(x_extent, y_extent)
    test_3D_data =  get_3d_test_data(test_2D_coords, test_time_stamp)

    # Predict and show map
    pred, pred_grid, std_pred = predict_map(gpr, test_3D_data, test_grid.shape[:2])
    show_map(pred_grid, [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]], lwc_unit, lwc_extent, test_time_stamp)
