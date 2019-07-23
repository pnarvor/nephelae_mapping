from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import random
from scipy import ndimage
import pickle
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import MFDataset
from matplotlib import animation
import matplotlib
from nephelae_mapping.test.util import load_pickle


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
    pred_var = plt.imshow(pred_grid.T, origin='lower', extent=xy_extent, cmap=plt.cm.viridis)

    cbar = plt.colorbar(fraction=0.0471, pad=0.01)
    cbar.ax.set_title(data_unit, pad=14)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.clim(data_extent[0], data_extent[1])
    plt.tight_layout()
    return fig, pred_var


if __name__ == "__main__":

    coord = load_pickle("coord")
    lwc_data = load_pickle("lwc_data")

    #coord, lwc_data = load_samples("coord.pickle", "lwc_data.pickle")

    lwc_extent = [0, 4.5e-4] # range of lwc
    lwc_unit = "kg a/kg w"
    noise_std = 1e-6
    zero_thresholding = True

    x_extent = load_pickle("xExtent")
    y_extent = load_pickle("yExtent")
    #x_extent, y_extent = load_window_size("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc", [60,150], [164,186])

    # Plot GT points
    save_sample_points = True
    plot_sampled_points(coord, lwc_data, x_extent, y_extent, lwc_extent, lwc_unit, save_fig = save_sample_points)

    # Add noise to data
    lwc_data =  noisy_samples(lwc_data, noise_std, zero_thresholding)

    # GPR Params
    sigma_f = 1e-4
    sigma_f_bounds = (1e-2, 9)
    lengthscale = [50, 50, 20]
    lengthscale_bounds = [(5, 500), (5, 500), (1, 300)]

    # Define Kernel
    kernel = RBF_kernel(sigma_f, sigma_f_bounds, lengthscale, lengthscale_bounds)
    test_2D_coords, test_grid = get_2d_test_grid(x_extent, y_extent)

    ##======1. Fit GPR on all gathered points and predict the whole map at last second!========
    # gpr = fit_gpr(coord, lwc_data, kernel, noise_std, 0)
    #
    # # Test Data
    # test_time_stamp = coord[-1, 2] # second(timestamp) of last sample
    # test_2D_coords, test_grid = get_2d_test_grid(x_extent, y_extent)
    # test_3D_data =  get_3d_test_data(test_2D_coords, test_time_stamp)
    #
    # # Predict and show map
    # pred, pred_grid, std_pred = predict_map(gpr, test_3D_data, test_grid.shape[:2])
    # show_map(pred_grid, [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]], lwc_unit, lwc_extent, test_time_stamp)

    ##======2. Fit GPR each second and predict map/s ========
    anim_save = False
    if(anim_save):
        matplotlib.use("Agg")
        plt.rcParams['animation.ffmpeg_path'] = '/home/dlohani/miniconda3/envs/nephelae/bin/ffmpeg'

    fig, pred_var = show_map( np.zeros(test_grid.shape[:2]), [x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]], lwc_unit, lwc_extent, 0)

    def init():
        #do nothing
        pass

    def update(t):
        gpr = fit_gpr(coord[:t+1], lwc_data[:t+1], kernel, noise_std, 0)
        test_time_stamp = coord[t, 2]  # second(timestamp) of last sample
        test_3D_data = get_3d_test_data(test_2D_coords, test_time_stamp)

        pred, pred_grid, std_pred = predict_map(gpr, test_3D_data, test_grid.shape[:2])
        pred_var.set_data(pred_grid.T)
        plt.title("Prediction at t = %ds, lengthscales=(%dm, %dm, %ds), sigma_f=%.2e %s" % (test_time_stamp, lengthscale[0], lengthscale[1], lengthscale[2], sigma_f, lwc_unit))
        plt.scatter(coord[t, 0], coord[t, 1], c="white", marker="x", s=15)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(len(lwc_data)),
        init_func=init,
        repeat=False,
        interval=200,
        )

    if(anim_save):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('predict_3d.mp4', writer=writer)
    else:
        plt.show(block=False)
    print("done")