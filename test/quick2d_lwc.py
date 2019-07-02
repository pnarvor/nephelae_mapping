from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import numpy as np
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
import random
import cloud
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import pickle
from scipy import ndimage
plt.rc('font', size=13) #default fontsize

def csec_cplot(var_data, t_index, z_index, tr, zr, xr, yr, ext, colors=None, alpha=None, new_fig=1, type="nc", show=0):
    # if(new_fig==1):
    #     plt.figure()
    # plt.xlabel("x coordinate(km)")
    # plt.ylabel("y coordinate(km)")
    data = var_data[t_index, z_index, ext[0]:ext[1], ext[2]:ext[3]]
    # plt.title("Cloud Cross Section at z={:.3f}km, t={}s".format(zr[z_index], tr[t_index]))

    if(type=="c"):
        cmap="Greys"
        if(colors!=None):
            cmap=None
        plt.contour(data.T, origin='lower', extent=[xr[ext[0]], xr[ext[1]-1], yr[ext[2]], yr[ext[3]-1]], colors=colors, cmap=cmap, alpha=alpha)
    elif(show!=0):
        plt.imshow(data.T, origin='lower', extent=[xr[ext[0]], xr[ext[1]-1], yr[ext[2]], yr[ext[3]-1]],vmin=abs(data.min()), vmax=abs(data.max()),cmap="viridis",alpha=alpha)
        # cbar = plt.colorbar(fraction=0.046, pad=0.04)
        # cbar.set_label('kg/kg')
    return data

def cloud_essential_data_extraction(y_pred, std_pred, type, std_pred_factor=1, thres=1e-5):
    if(type=="l"):
        data = y_pred - std_pred_factor * std_pred
    elif(type=="h"):
        data = y_pred + std_pred_factor * std_pred
    else:
        data = 0
        print("Wrong Type Parameter!!!")
    data_grid = np.reshape(data, (-1, 70)).T
    data_b = data
    data_b[data_b < thres] = 0  # thresholding
    data_b = np.reshape(data_b, (-1, 70)).T
    return data_b, data_grid

def mean_prior(y, sigma_blur = 7, type= 1, circ=[35, 35, 20]):
    mf = ndimage.gaussian_filter(y, sigma=sigma_blur) #Blurring
    if(type==2): # circular prior
        mf = circluar_prior(mf, circ[0], circ[1], circ[2])
    return mf, mf.flatten().reshape(4900,1)

def circluar_prior(mf, a, b, r):
    h,w = mf.shape
    dum_mf = np.zeros((h,w))
    for j in range(w):
        for i in range(h):
            if ((i - a) ** 2 + (j - b) ** 2) <= r ** 2:
                dum_mf[j][i] = mf[j][i]
    return dum_mf

def circluar_trajectory(mf, a, b, r, EPS):
    h,w = mf.shape
    data = []
    for j in range(w):
        for i in range(h):
            if abs((i - a) ** 2 + (j - b) ** 2 - r ** 2) <= EPS ** 2:
                data= np.append(data, mf[j][i])
    return data

def same_dist_sample_points(arr, sample_gap):
    num_samples = int(len(arr) / sample_gap)
    indices = [(i * sample_gap)-1 for i in range(1,num_samples+1)]
    return arr[indices], indices

def choose_random_sample(arr, per):
    n_samples = int(per*len(arr)/100)
    indices = random.sample(range(len(arr)), n_samples)
    return arr[indices], indices

def get_color_nd_norm(type="lwc", c=[36, 5, 100, 17], b=[-10e-5,35e-5,1e-5]):
    if(type=="zwind"):
        c = [70, 2, 100, 8]
        b = [-4.48, 7.75, 1e-1]
    colors1 = plt.cm.PuRd(np.linspace(0., 1, c[0]))
    colors2 = plt.cm.Greys(np.linspace(0., 1, c[1]))
    colors3 = plt.cm.viridis(np.linspace(0, 1, c[2]))
    colors4 = plt.cm.Oranges(np.linspace(0., 1, 35))[c[3]:,:]
    colors = np.vstack((colors1, colors2, colors3, colors4))
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap_cloud = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(b[0], b[1], b[2]) #lwc rct bounds to find, zwind maybe -3.48 to 6.75
    idx = np.searchsorted(bounds, 0)
    bounds = np.insert(bounds, idx, 0)
    norm_cloud = mcolors.BoundaryNorm(bounds, cmap_cloud.N)
    return cmap_cloud, norm_cloud

def criss_cross(Xt_grid, y, sigma_n, i, mf=[]):

    if(i==0):
        x, XTi = same_dist_sample_points(Xt_grid[:, 35], 2)
        ytt = y[XTi, 35] + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            ytt = ytt - mf[XTi, 35] #mean func
    elif(i==1):
        x, XTi = same_dist_sample_points(Xt_grid[35, :], 2)
        ytt = y[35, XTi] + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            ytt = ytt - mf[35, XTi] #mean func
    elif(i==2):
        y_rd= np.diag(y)
        y_rd,y_rdi = same_dist_sample_points(y_rd,2)
        x1_rd= np.diag(Xt_grid[:,:,0])
        x1_rd= x1_rd[y_rdi]
        x2_rd= np.diag(Xt_grid[:,:,1])
        x2_rd= x2_rd[y_rdi]
        x=np.vstack((x1_rd, x2_rd)).T
        ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf) #mean func
            ytt = ytt - y_rdmf[y_rdi]
    elif(i==3):
        y_rd= np.diag(np.fliplr(y))
        y_rd,y_rdi = same_dist_sample_points(y_rd,2)
        x1_rd= np.diag(np.fliplr(Xt_grid[:,:,0]))
        x1_rd= x1_rd[y_rdi]
        x2_rd= np.diag(np.fliplr(Xt_grid[:,:,1]))
        x2_rd= x2_rd[y_rdi]
        x=np.vstack((x1_rd, x2_rd)).T
        ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(np.fliplr(mf)) #mean func
            ytt = ytt - y_rdmf[y_rdi]
    elif (i == 4):
        far_diag = 20
        y_rd = np.diag(y, k=far_diag)
        y_rd, y_rdi = same_dist_sample_points(y_rd, 2)
        x1_rd = np.diag(Xt_grid[:, :, 0], k=far_diag)
        x1_rd = x1_rd[y_rdi]
        x2_rd = np.diag(Xt_grid[:, :, 1], k=far_diag)
        x2_rd = x2_rd[y_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf, k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
    elif (i == 5):
        far_diag = 20
        y_rd = np.diag(np.fliplr(y), k=far_diag)
        y_rd, y_rdi = same_dist_sample_points(y_rd, 2)
        x1_rd = np.diag(np.fliplr(Xt_grid[:, :, 0]), k=far_diag)
        x1_rd = x1_rd[y_rdi]
        x2_rd = np.diag(np.fliplr(Xt_grid[:, :, 1]), k=far_diag)
        x2_rd = x2_rd[y_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(np.fliplr(mf), k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
    elif (i == 6):
        far_diag = -20
        y_rd = np.diag(y, k=far_diag)
        y_rd, y_rdi = same_dist_sample_points(y_rd, 2)
        x1_rd = np.diag(Xt_grid[:, :, 0], k=far_diag)
        x1_rd = x1_rd[y_rdi]
        x2_rd = np.diag(Xt_grid[:, :, 1], k=far_diag)
        x2_rd = x2_rd[y_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf, k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
    elif (i == 7):
        far_diag = -20
        y_rd = np.diag(np.fliplr(y), k=far_diag)
        y_rd, y_rdi = same_dist_sample_points(y_rd, 2)
        x1_rd = np.diag(np.fliplr(Xt_grid[:, :, 0]), k=far_diag)
        x1_rd = x1_rd[y_rdi]
        x2_rd = np.diag(np.fliplr(Xt_grid[:, :, 1]), k=far_diag)
        x2_rd = x2_rd[y_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(np.fliplr(mf), k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
    return x, ytt

def circular(Xt_grid, y, sigma_n, r, eps=4, a=34, b=34, mf=[]):
    # if(r in (28,24)):
    #     eps=2.3
    # elif (r in (18, 15,12)):
    #     eps = 2.5
    # elif (r in (9, 4)):
    #     eps = 2.6
    y_rd = circluar_trajectory(y, a, b, r, eps)
    x1_rd = circluar_trajectory(Xt_grid[:, :, 0], a, b, r, eps)
    x2_rd = circluar_trajectory(Xt_grid[:, :, 1], a, b, r, eps)
    x1_rd, xi = same_dist_sample_points(x1_rd, 3)
    x2_rd = x2_rd[xi]
    y_rd = y_rd[xi]
    x = np.vstack((x1_rd, x2_rd)).T
    if (r == 24):
        x = np.delete(x, [2, 30, 35], axis=0)
        y_rd = np.delete(y_rd, [2, 30, 35], axis=0)
    elif (r == 19):
        x = np.delete(x, [4, 11, 12, 19, 22, 25, 26, 33], axis=0)
        y_rd = np.delete(y_rd, [4, 11, 12, 19, 22, 25, 26, 33], axis=0)
    elif (r == 14):
        x = np.delete(x, [3, 7, 11, 13, 15, 21, 23, 29, 26, 33, 25], axis=0)
        y_rd = np.delete(y_rd, [3, 7, 11, 13, 15, 21, 23, 29, 26, 33, 25], axis=0)
    elif (r == 9):
        x = np.delete(x, [2, 3, 4, 5, 9, 15, 21, 24, 29, 12, 13, 31], axis=0)
        y_rd = np.delete(y_rd, [2, 3, 4, 5, 9, 15, 21, 24, 29, 12, 13, 31], axis=0)

    ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
    # if (mf.size != 0):
    #     #ytt = ytt - mf[XTi, 35]  # mean func
    return x, ytt

def com(data, coord):
    mass_sum = np.sum(data)
    return np.sum(data * coord[:,0])/ mass_sum, np.sum(data * coord[:,1])/ mass_sum


def run_gp_circular(mf, save_path=""):
    # plt.ioff() #no image display
    train_sam_num = 0
    XT = np.empty((0, 2), float)
    yT = []
    lmlt = []
    len_scale = []
    sigma_f = []
    mse = []
    num_points = []
    sigma_n = 1e-6  # lwc 1e-6 zwind 0.2
    len_scale = np.append(len_scale, 0.02)
    len_scale_bounds = (1e-2, 5e-1)
    sigma_f = np.append(sigma_f, 2e-4)  # lwc 2e-4 zwind 2
    sigma_f_bounds = (1e-6, 3e-4)  # lwc 1e-6, 3e-4 zwind 1e-2, 9
    if (mf.size != 0):
        type = mf[0]
        sigma_blur = mf[1]
        circ = mf[2:]
        mf, mf_flat = mean_prior(y, sigma_blur, type, circ)
    #  if(traj == "circular"):
    for r in (34, 29, 24, 19, 14, 9):  # circ traj
        x, ytt = circular(Xt_grid, y, sigma_n, r)  # TODO: Mean prior with circular!!
        XT = np.append(XT, x, axis=0)
        ytt = ytt.clip(min=0)  # lwc
        yT = np.append(yT, ytt)
        kernel = C(constant_value=sigma_f[-1] ** 2,
                   constant_value_bounds=(sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
                 RBF(length_scale=len_scale[-1], length_scale_bounds=len_scale_bounds)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=50)
        gp.fit(XT, yT.reshape(-1, 1))
        sigma = np.sqrt(gp.kernel_.k1.get_params()['constant_value'])
        l = gp.kernel_.k2.get_params()['length_scale']
        lml = gp.log_marginal_likelihood()  # zwind -gp.log_marginal_likelihood()
        len_scale = np.append(len_scale, l)
        sigma_f = np.append(sigma_f, sigma)
        lmlt = np.append(lmlt, lml)
        print("length_scale : ", l)
        print("sigma_f : ", sigma)
        print("log marginal likelihood: ", lml)
        train_sam_num += len(x)
        num_points = np.append(num_points, train_sam_num)
        train_per = train_sam_num / float(4900)
        y_pred, std_pred = gp.predict(Xt, return_std=True)
        std_pred = std_pred.reshape(4900, 1)
        if (mf.size != 0):
            y_pred = mf_flat + y_pred  # mean func
        y_error = yt - y_pred[:, 0]
        com_pred_x, com_pred_y = com(y_pred[:, 0], Xt)
        y_mse = np.sqrt(np.mean(np.square(y_error)))
        mse = np.append(mse, y_mse)
        y_pred_grid = np.reshape(y_pred, (-1, 70)).T

        y_pred_b = y_pred  # lwc
        y_pred_b[y_pred_b < 1e-5] = 0  # lwc thresholding
        y_pred_b = np.reshape(y_pred_b, (-1, 70)).T  # lwc
        # lwc border
        std_pred_factor = 1
        lower_border_b, lower_border_grid = cloud_essential_data_extraction(y_pred, std_pred, type="l")  # lower border
        higher_border_b, higher_border_grid = cloud_essential_data_extraction(y_pred, std_pred,
                                                                              type="h")  # higher border
        if (1):  # 1 or i==7 r==9
            plt.figure(figsize=(24, 6), dpi=62)
            plt.subplot(1, 4, 1, xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title('GT with %2d (%4.2f%%) Training Points' % (train_sam_num, train_per * 100), size=14)
            csec_cplot(lwc_cloud1, 5, 0, tr, zr, xr, yr, [30, 100, 60, 130], colors="black", alpha=0.3, type="c")
            plt.scatter(com_x, com_y, c="white", marker="x", s=80)
            plt.scatter(XT[:, 0], XT[:, 1], c="black", marker="x", s=8)
            plt.imshow(y.T, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest',
                       norm=norm_cloud)
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069))
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            cbar = plt.colorbar(fraction=0.0471, pad=0.01)  # zwind, format='%.1f')
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.subplot(1, 4, 2, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title("Prediction $l$: %dm, $\sigma_f$: %2.3e" % (l * 1000, sigma), size=14)  # lwc
            plt.contour(higher_border_b, origin='lower', extent=cloud_extent, colors="darkgray", levels=0)  # lwc
            plt.contour(lower_border_b, origin='lower', extent=cloud_extent, colors="darkgray", levels=0)  # lwc
            plt.contour(y_pred_b, origin='lower', extent=cloud_extent, colors="Black", levels=0)  # lwc
            # zwind csec_cplot(lwc_cloud1, 5, 0, tr, zr, xr, yr, [30, 100, 60, 130], colors="black", alpha=0.3, type="c")
            plt.scatter(com_pred_x, com_pred_y, c="white", marker="x", s=80)
            plt.imshow(y_pred_grid, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest',
                       norm=norm_cloud)
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069), visible=False)
            # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            cbar = plt.colorbar(fraction=0.0471, pad=0.01)  # zwind, format='%.1f')
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.subplot(1, 4, 3, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title("Predicted Standard Deviation", size=14)
            plt.imshow(np.reshape(std_pred, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=std_pred.min(),
                       vmax=std_pred.max(), cmap="jet")
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069), visible=False)
            # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.clim(0, 1.5e-4)  # zwind plt.clim(0, 2.25)
            cbar = plt.colorbar(fraction=0.0471, pad=0.01)  # zwind , format='%.1f')
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.subplot(1, 4, 4, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title("Error, RMSE: %2.2e" % (y_mse), size=14)
            plt.imshow(np.reshape(y_error, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=y_error.min(),
                       vmax=y_error.max(), cmap="jet")
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069), visible=False)
            # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.clim(-1e-4, 2.5e-4)  # zwind plt.clim(-2, 5)
            cbar = plt.colorbar(fraction=0.0471, pad=0.01)
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.tight_layout()
            # plt.show()
            if (save_path):
                plt.savefig(save_path + str(train_sam_num) + 'points.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale[1:] * 1000)
    plt.scatter(num_points, len_scale[1:] * 1000, c="black", marker="x")
    plt.text(num_points[0], len_scale[1] * 1000, round(len_scale[1] * 1000, 2), transform=plt.gca().transData,
             fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale[-1] * 1000, round(len_scale[-1] * 1000, 2), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired", fontsize=18)
    plt.ylabel("$l$ (m)", fontsize=18)
    plt.title("Evolution of lengthscale (Init: $l$= %dm, Bounds=(%dm, %dm))" % (
        len_scale[0] * 1000, len_scale_bounds[0] * 1000, len_scale_bounds[1] * 1000), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lengthscale.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, sigma_f[1:])
    plt.scatter(num_points, sigma_f[1:], c="black", marker="x")
    plt.text(num_points[0], sigma_f[1], "%.2e" % (sigma_f[1]), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], sigma_f[-1], "%.2e" % (sigma_f[-1]), transform=plt.gca().transData, fontsize=16,
             ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))  # lwc
    plt.xlabel("Number of data points acquired", fontsize=18)
    # plt.ylabel("$\sigma_f $(unit: m/s)", fontsize=16)
    plt.ylabel("$\sigma_f $(unit: kg water/kg air)", fontsize=16)  # lwc
    # plt.title("Evolution of Signal Standard Deviation (Init: $\sigma_f$: %3.2f, Bounds=(%3.2f, %3.2f))" % (
    #   sigma_f[0], sigma_f_bounds[0], sigma_f_bounds[1]), fontsize=18)
    plt.title("Evolution of Signal Standard Deviation (Init: $\sigma_f$: %1.0e, Bounds=(%1.0e, %1.0e))" % (
        sigma_f[0], sigma_f_bounds[0], sigma_f_bounds[1]), fontsize=18)  # lwc
    if (save_path):
        plt.savefig(save_path + 'out_std.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, lmlt)
    plt.scatter(num_points, lmlt, c="black", marker="x")
    plt.text(num_points[0], lmlt[0], round(lmlt[0], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], lmlt[-1], round(lmlt[-1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired", fontsize=18)
    plt.ylabel("LML Score", fontsize=18)
    plt.title("Evolution of Log Marginal Likelihood Score", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lml.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, mse)
    plt.scatter(num_points, mse, c="black", marker="x")
    plt.text(num_points[0], mse[0], "%.2e" % (mse[0]), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom',
             color='red', weight='bold')
    plt.text(num_points[-1], mse[-1], "%.2e" % (mse[-1]), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))  # lwc
    plt.xlabel("Number of data points acquired", fontsize=18)
    plt.ylabel("RMSE Score (unit: kg water/kg air)", fontsize=18)  # lwc
    # plt.ylabel("RMSE Score (unit: m/s)", fontsize=18)
    plt.title("Evolution of Root Mean Square Error", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'rmse.png')

def run_gp_criss_cross(mf, save_path=""):
     #plt.ioff() #no image display
    train_sam_num = 0
    XT = np.empty((0, 2), float)
    yT = []
    lmlt = []
    len_scale = []
    sigma_f = []
    mse = []
    num_points = []
    sigma_n = 1e-6  # lwc 1e-6 zwind 0.2
    len_scale = np.append(len_scale, 0.02)
    len_scale_bounds = (1e-2, 5e-1)
    sigma_f = np.append(sigma_f, 2e-4)  # lwc 2e-4 zwind 2
    sigma_f_bounds = (1e-6, 3e-4)  # lwc 1e-6, 3e-4 zwind 1e-2, 9
    if (mf.size != 0):
        type = mf[0]
        sigma_blur = mf[1]
        circ = mf[2:]
        mf, mf_flat = mean_prior(y, sigma_blur, type, circ)
        plt.figure()
        plt.title("Prior Mean on LWC")
        plt.xlabel("x coordinate(km)")
        plt.ylabel("y coordinate(km)")
        plt.imshow(mf.T, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest', norm=norm_cloud)
        plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01)  # zwind, format='%.1f')
        cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        if (save_path):
            plt.savefig(save_path + 'prior_mean.png')

    #For criss cross traj
    for i in range(8):
        x, ytt = criss_cross(Xt_grid, y, sigma_n, i, mf)
        XT = np.append(XT, x, axis=0)
        ytt = ytt.clip(min=0) #lwc
        yT = np.append(yT, ytt)
        kernel = C(constant_value=sigma_f[-1] ** 2,
                   constant_value_bounds=(sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
                 RBF(length_scale=len_scale[-1], length_scale_bounds=len_scale_bounds)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=50)
        gp.fit(XT, yT.reshape(-1, 1))
        sigma = np.sqrt(gp.kernel_.k1.get_params()['constant_value'])
        l = gp.kernel_.k2.get_params()['length_scale']
        lml = gp.log_marginal_likelihood()# zwind -gp.log_marginal_likelihood()
        len_scale = np.append(len_scale, l)
        sigma_f = np.append(sigma_f, sigma)
        lmlt = np.append(lmlt, lml)
        print("length_scale : ", l)
        print("sigma_f : ", sigma)
        print("log marginal likelihood: ", lml)
        train_sam_num += len(x)
        num_points = np.append(num_points, train_sam_num)
        train_per = train_sam_num / float(4900)
        y_pred, std_pred = gp.predict(Xt, return_std=True)
        std_pred = std_pred.reshape(4900, 1)
        if (mf.size != 0):
            y_pred = mf_flat + y_pred  # mean func
        y_error = yt - y_pred[:, 0]
        com_pred_x, com_pred_y = com(y_pred[:, 0], Xt)
        y_mse = np.sqrt(np.mean(np.square(y_error)))
        mse = np.append(mse, y_mse)
        y_pred_grid = np.reshape(y_pred, (-1, 70)).T

        y_pred_b= y_pred #lwc
        y_pred_b[y_pred_b<1e-5]=0 #lwc thresholding
        y_pred_b = np.reshape(y_pred_b, (-1, 70)).T #lwc
        # lwc border
        std_pred_factor = 1
        lower_border_b, lower_border_grid = cloud_essential_data_extraction(y_pred, std_pred, type="l") #lower border
        higher_border_b, higher_border_grid = cloud_essential_data_extraction(y_pred, std_pred, type="h") #higher border
        if (1):  # 1 or i==7 r==9
            plt.figure(figsize=(24, 6), dpi=62)
            plt.subplot(1, 4, 1, xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title('GT with %2d (%4.2f%%) Training Points' % (train_sam_num, train_per * 100), size=14)
            csec_cplot(lwc_cloud1, 5, 0, tr, zr, xr, yr, [30, 100, 60, 130], colors="black", alpha=0.3, type="c")
            plt.scatter(com_x, com_y, c="white", marker="x", s=80)
            plt.scatter(XT[:, 0], XT[:, 1], c="black", marker="x", s=8)
            plt.imshow(y.T, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest', norm=norm_cloud)
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069))
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            cbar = plt.colorbar(fraction=0.0471, pad=0.01) # zwind, format='%.1f')
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14) #zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.subplot(1, 4, 2, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title("Prediction $l$: %dm, $\sigma_f$: %2.3e" % (l*1000, sigma), size=14) #lwc
            plt.contour(higher_border_b, origin='lower', extent=cloud_extent, colors="darkgray", levels=0) #lwc
            plt.contour(lower_border_b, origin='lower', extent=cloud_extent, colors="darkgray", levels=0) #lwc
            plt.contour(y_pred_b, origin='lower', extent=cloud_extent, colors="Black", levels=0)  #lwc
            #zwind csec_cplot(lwc_cloud1, 5, 0, tr, zr, xr, yr, [30, 100, 60, 130], colors="black", alpha=0.3, type="c")
            plt.scatter(com_pred_x, com_pred_y, c="white", marker="x", s=80)
            plt.imshow(y_pred_grid, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest',
                       norm=norm_cloud)
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069), visible=False)
            #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            cbar = plt.colorbar(fraction=0.0471, pad=0.01)#zwind, format='%.1f')
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.subplot(1, 4, 3, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title("Predicted Standard Deviation", size=14)
            plt.imshow(np.reshape(std_pred, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=std_pred.min(),
                       vmax=std_pred.max(), cmap="jet")
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069), visible=False)
            #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.clim(0, 1.5e-4) #zwind plt.clim(0, 2.25)
            cbar = plt.colorbar(fraction=0.0471, pad=0.01) #zwind , format='%.1f')
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.subplot(1, 4, 4, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
                .set_title("Error, RMSE: %2.2e" % (y_mse), size=14)
            plt.imshow(np.reshape(y_error, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=y_error.min(),
                       vmax=y_error.max(), cmap="jet")
            plt.xticks(np.arange(xr[30], xr[100] + 0.069, 2 * 0.069))
            plt.yticks(np.arange(yr[60], yr[130] + 0.069, 2 * 0.069), visible=False)
            #plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
            plt.clim(-1e-4, 2.5e-4) #zwind plt.clim(-2, 5)
            cbar = plt.colorbar(fraction=0.0471, pad=0.01)
            cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)  # zwind cbar.ax.set_title('   m/s', fontsize=15, pad=7)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

            plt.tight_layout()
            #plt.show()
            if (save_path):
                plt.savefig(save_path + str(train_sam_num) + 'points.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale[1:] * 1000)
    plt.scatter(num_points, len_scale[1:] * 1000, c="black", marker="x")
    plt.text(num_points[0], len_scale[1] * 1000, round(len_scale[1] * 1000, 2), transform=plt.gca().transData, fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale[-1] * 1000, round(len_scale[-1] * 1000, 2), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired", fontsize=18)
    plt.ylabel("$l$ (m)", fontsize=18)
    plt.title("Evolution of lengthscale (Init: $l$= %dm, Bounds=(%dm, %dm))" % (
        len_scale[0] * 1000, len_scale_bounds[0] * 1000, len_scale_bounds[1] * 1000), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lengthscale.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, sigma_f[1:])
    plt.scatter(num_points, sigma_f[1:], c="black", marker="x")
    plt.text(num_points[0], sigma_f[1], "%.2e"%(sigma_f[1]), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], sigma_f[-1], "%.2e"%(sigma_f[-1]), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e')) #lwc
    plt.xlabel("Number of data points acquired", fontsize=18)
    #plt.ylabel("$\sigma_f $(unit: m/s)", fontsize=16)
    plt.ylabel("$\sigma_f $(unit: kg water/kg air)", fontsize=16) #lwc
    #plt.title("Evolution of Signal Standard Deviation (Init: $\sigma_f$: %3.2f, Bounds=(%3.2f, %3.2f))" % (
     #   sigma_f[0], sigma_f_bounds[0], sigma_f_bounds[1]), fontsize=18)
    plt.title("Evolution of Signal Standard Deviation (Init: $\sigma_f$: %1.0e, Bounds=(%1.0e, %1.0e))"% (
        sigma_f[0], sigma_f_bounds[0], sigma_f_bounds[1]), fontsize=18) #lwc
    if (save_path):
        plt.savefig(save_path + 'out_std.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, lmlt)
    plt.scatter(num_points, lmlt, c="black", marker="x")
    plt.text(num_points[0], lmlt[0], round(lmlt[0], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], lmlt[-1], round(lmlt[-1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired", fontsize=18)
    plt.ylabel("LML Score", fontsize=18)
    plt.title("Evolution of Log Marginal Likelihood Score", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lml.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, mse)
    plt.scatter(num_points, mse, c="black", marker="x")
    plt.text(num_points[0], mse[0], "%.2e"%(mse[0]), transform=plt.gca().transData, fontsize=16, ha='center', va='bottom',
             color='red', weight='bold')
    plt.text(num_points[-1], mse[-1], "%.2e"%(mse[-1]), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e')) #lwc
    plt.xlabel("Number of data points acquired", fontsize=18)
    plt.ylabel("RMSE Score (unit: kg water/kg air)", fontsize=18) #lwc
    #plt.ylabel("RMSE Score (unit: m/s)", fontsize=18)
    plt.title("Evolution of Root Mean Square Error", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'rmse.png')

# path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
# mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
#           for minute in range(1, 16) #orig 60
#           for second in range(1, 61)]
# atm = MesoNHAtmosphere(mfiles, 1)
#
# lwc_data1 = atm.data['RCT'][449:455,89:90,60:200,110:250]  #89:90 means only 1 height 1.125 km85:123 range of z
# zwind_data1 = atm.data['WT'][449:455,89:90,60:200,110:250] #85:123 range of z
#
# ids1,counter1,clouds1=cloud.cloud_segmentation(lwc_data1)
# clouds1=list(set(clouds1.values()))
# length_point_clds = np.ndarray((0,1))
# for each_cloud in clouds1:
#     temp = len(each_cloud.points)
#     length_point_clds = np.vstack((length_point_clds,temp))
#
# sorted_indices = length_point_clds[:,0].argsort()[::-1] # clouds sorted acc to #cloud_points
# cloud1 = clouds1[sorted_indices[0]] #Biggest cloud
#
# cloud1.calculate_attributes(lwc_data1,zwind_data1) #zwind also
# #cloud1.calculate_attributes(lwc_data1)
#
# lwc_cloud1 = np.zeros(lwc_data1.shape)
# for point in cloud1.points:
#     lwc_cloud1[point] = 1
# del clouds1
# all_Zs=atm.data["VLEV"][:,0,0]

# Dumping
# pickle_out = open("lwc_cloud.pickle","wb")
# pickle.dump(lwc_cloud1, pickle_out)
# pickle_out.close()

# Loading
# pickle_in = open("lwc_data1_2d.pickle","rb")
# lwc_data1 = pickle.load(pickle_in)
# pickle_in = open("zwind_data1_2d.pickle","rb")
# zwind_data1 = pickle.load(pickle_in)
# pickle_in = open("lwc_cloud1_2d.pickle","rb")
# lwc_cloud1 = pickle.load(pickle_in)
# pickle_in = open("all_Zs_2d.pickle","rb")
# all_Zs = pickle.load(pickle_in)

pickle_in = open("lwc_data1_2d.pickle","rb")
lwc_data1 = pickle.load(pickle_in, encoding='latin1')
pickle_in = open("zwind_data1_2d.pickle","rb")
zwind_data1 = pickle.load(pickle_in, encoding='latin1')
pickle_in = open("lwc_cloud1_2d.pickle","rb")
lwc_cloud1 = pickle.load(pickle_in, encoding='latin1')
pickle_in = open("all_Zs_2d.pickle","rb")
all_Zs = pickle.load(pickle_in, encoding='latin1')

xr =np.arange(0.005 + 60*0.01, 0.005 + 200*0.01,0.01)
yr= np.arange(0.005 + 110*0.01, 0.005 + 250*0.01,0.01)
zr = all_Zs[89:90] #85:123 range of z
tr = np.arange(449,455)

y = csec_cplot(lwc_data1, 5, 0, tr, zr, xr, yr,[30,100,60,130])
x1 = xr[30:100]
x2 = yr[60:130]

# data for test on whole patch 70X70
Xt_grid = np.array(np.meshgrid(x1,x2)).T
Xt = Xt_grid.reshape(-1,2)
yt = y.flatten()
com_x, com_y = com(yt, Xt)
cloud_extent = [xr[30], xr[99], yr[60], yr[129]]

cmap_cloud, norm_cloud = get_color_nd_norm() #lwc
mf = np.array([]) # zero mean prior
#mf = np.array([2, 7, 35, 35, 20])
# [0] -> 1 whole, 2 circular
# [1] -> sigma_blur
# [2][3][4] -> centre + radius of circular prior
save_path =""

# EXP 1 (without prior, criss cross trajectory)
# mf = np.array([]) # zero mean prior
# run_gp_criss_cross(mf, save_path)

# EXP 2 (without prior, circular trajectory)
# mf = np.array([]) # zero mean prior
# run_gp_circular(mf, save_path)

# EXP 3 (circular prior)
#mf = np.array([2, 7, 35, 35, 20])
# # [0] -> 1 whole, 2 circular
# # [1] -> sigma_blur
# # [2][3][4] -> centre + radius of circular prior
# run_gp_criss_cross(mf, save_path)

# EXP 4 (prior on whole image)
#mf = np.array([1, 7, 35, 35, 20])
# # [0] -> 1 whole, 2 circular
# # [1] -> sigma_blur
# # [2][3][4] -> centre + radius of circular prior
# run_gp_criss_cross(mf, save_path)
