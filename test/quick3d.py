from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C #, WhiteKernel as W
import numpy as np
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere
import matplotlib.pyplot as plt
import random
#import cloud
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

def criss_cross(Xt_grid, data_interest, sigma_n, i, t_start, mf=[], ind = 35):
    if(i==0):
        x, XTi = same_dist_sample_points(Xt_grid[:, ind], 2)
        ytt = []
        for t in range(len(XTi)):
            ytt = np.append(ytt, data_interest[t_start + t, 0, XTi[t], ind])
        for t in range(len(XTi)):
            ytt = np.append(ytt, data_interest[t_start +3+ t, 0, XTi[t], ind])
        for t in range(len(XTi)):
            ytt = np.append(ytt, data_interest[t_start +6+ t, 0, XTi[t], ind])
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(ytt))
        if (mf.size != 0):
            ytt = ytt - mf[XTi, ind] #TODO mean func
        x = np.concatenate((np.concatenate((x,x,x)), np.concatenate((tr[t_start: t_start+len(XTi)],tr[t_start+3: t_start+3+len(XTi)], tr[t_start+6: t_start+6+len(XTi)])).reshape(-1, 1)), axis=1)
    elif(i==1):
        x, XTi = same_dist_sample_points(Xt_grid[ind, :], 2)
        ytt = []
        for t in range(len(XTi)):
            ytt = np.append(ytt, data_interest[t_start+ t, 0, ind, XTi[t]])
        for t in range(len(XTi)):
            ytt = np.append(ytt, data_interest[t_start +3+ t, 0, ind, XTi[t]])
        for t in range(len(XTi)):
            ytt = np.append(ytt, data_interest[t_start + 6 + t, 0, ind, XTi[t]])
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(ytt))
        if (mf.size != 0):
            ytt = ytt - mf[35, XTi] #TODO mean func
        x = np.concatenate((np.concatenate((x,x,x)), np.concatenate((tr[t_start: t_start+len(XTi)],tr[t_start+3: t_start+3+len(XTi)], tr[t_start+6: t_start+6+len(XTi)])).reshape(-1, 1)), axis=1)
    elif(i==2):
        x1_rd = np.diag(Xt_grid[:, :, 0])
        x1_rd, x1_rdi = same_dist_sample_points(x1_rd, 2)
        x2_rd = np.diag(Xt_grid[:, :, 1])
        x2_rd = x2_rd[x1_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt=[]
        for t in range(len(x1_rdi)):
            ytt = np.append(ytt, data_interest[t_start+ t, 0, x1_rdi[t], x1_rdi[t]])
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf) #TODO mean func
            #ytt = ytt - y_rdmf[y_rdi]
        x = np.concatenate((x, tr[t_start: t_start + len(x)].reshape(-1, 1)), axis=1)
    elif(i==3):
        x1_rd = np.diag(np.fliplr(Xt_grid[:, :, 0]))
        x1_rd, x1_rdi = same_dist_sample_points(x1_rd, 2)
        x2_rd = np.diag(np.fliplr(Xt_grid[:, :, 1]))
        x2_rd = x2_rd[x1_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = []
        for t in range(len(x1_rdi)):
            y_rd = np.diag(np.fliplr(data_interest[t_start + t, 0]))[x1_rdi[t]]
            ytt = np.append(ytt, y_rd)
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(np.fliplr(mf)) #TODO mean func
            ytt = ytt - y_rdmf[y_rdi]
        x = np.concatenate((x, tr[t_start: t_start + len(x)].reshape(-1, 1)), axis=1)
    elif (i == 4):
        far_diag = 20
        x1_rd = np.diag(Xt_grid[:, :, 0], k=far_diag)
        x1_rd, x1_rdi = same_dist_sample_points(x1_rd, 2)
        x2_rd = np.diag(Xt_grid[:, :, 1], k=far_diag)
        x2_rd = x2_rd[x1_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = []
        for t in range(len(x1_rdi)):
            y_rd = np.diag(data_interest[t_start + t, 0], k=far_diag)[x1_rdi[t]]
            ytt = np.append(ytt, y_rd)
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf, k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
        x = np.concatenate((x, tr[t_start: t_start + len(x)].reshape(-1, 1)), axis=1)
    elif (i == 5):
        far_diag = 20
        x1_rd = np.diag(np.fliplr(Xt_grid[:, :, 0]), k=far_diag)
        x1_rd, x1_rdi = same_dist_sample_points(x1_rd, 2)
        x2_rd = np.diag(np.fliplr(Xt_grid[:, :, 1]), k=far_diag)
        x2_rd = x2_rd[x1_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = []
        for t in range(len(x1_rdi)):
            y_rd = np.diag(np.fliplr(data_interest[t_start + t, 0]), k=far_diag)[x1_rdi[t]]
            ytt = np.append(ytt, y_rd)
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf, k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
        x = np.concatenate((x, tr[t_start: t_start + len(x)].reshape(-1, 1)), axis=1)
    elif (i == 6):
        far_diag = -20
        x1_rd = np.diag(Xt_grid[:, :, 0], k=far_diag)
        x1_rd, x1_rdi = same_dist_sample_points(x1_rd, 2)
        x2_rd = np.diag(Xt_grid[:, :, 1], k=far_diag)
        x2_rd = x2_rd[x1_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = []
        for t in range(len(x1_rdi)):
            y_rd = np.diag(data_interest[t_start + t, 0], k=far_diag)[x1_rdi[t]]
            ytt = np.append(ytt, y_rd)
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf, k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
        x = np.concatenate((x, tr[t_start: t_start + len(x)].reshape(-1, 1)), axis=1)
    elif (i == 7):
        far_diag = -20
        x1_rd = np.diag(np.fliplr(Xt_grid[:, :, 0]), k=far_diag)
        x1_rd, x1_rdi = same_dist_sample_points(x1_rd, 2)
        x2_rd = np.diag(np.fliplr(Xt_grid[:, :, 1]), k=far_diag)
        x2_rd = x2_rd[x1_rdi]
        x = np.vstack((x1_rd, x2_rd)).T
        ytt = []
        for t in range(len(x1_rdi)):
            y_rd = np.diag(np.fliplr(data_interest[t_start + t, 0]), k=far_diag)[x1_rdi[t]]
            ytt = np.append(ytt, y_rd)
        ytt = ytt + sigma_n ** 2 * np.random.randn(len(x))
        if (mf.size != 0):
            y_rdmf = np.diag(mf, k=far_diag)  # mean func
            ytt = ytt - y_rdmf[y_rdi]
        x = np.concatenate((x, tr[t_start: t_start + len(x)].reshape(-1, 1)), axis=1)
    return x, ytt

def circular(Xt_grid, y, sigma_n, r, eps=2.2, a=34, b=34, mf=[]):
    if(r in (28,24)):
        eps=2.3
    elif (r in (18, 15,12)):
        eps = 2.5
    elif (r in (9, 4)):
        eps = 2.6
    y_rd = circluar_trajectory(y, a, b, r, eps)
    y_rd, y_rdi = choose_random_sample(y_rd, 80)
    x1_rd = circluar_trajectory(Xt_grid[:, :, 0], a, b, r, eps)
    x1_rd = x1_rd[y_rdi]
    x2_rd = circluar_trajectory(Xt_grid[:, :, 1], a, b, r, eps)
    x2_rd = x2_rd[y_rdi]
    x = np.vstack((x1_rd, x2_rd)).T
    ytt = y_rd + sigma_n ** 2 * np.random.randn(len(x))
    # if (mf.size != 0):
    #     #ytt = ytt - mf[XTi, 35]  # mean func
    return x, ytt

def run_gp3D(save_path, data_interest):

    # * GP on 3d data with 3d kernel lx,ly,lt
    # Fly 3 drones on back to back (3 seconds apart)

    train_sam_num = 0
    XT = np.empty((0, 3), float)
    yT = []
    lmlt = []
    sigma_f = []
    sigma_f_x = []
    mse = []
    num_points = []
    len_scale_x = []
    len_scale_y = []
    len_scale_t = []
    sigma_n = 0.2  # lwc 1e-6
    sigma_f = np.append(sigma_f, 12)  # lwc 4 #
    sigma_f_x = np.append(sigma_f_x, 12)
    sigma_f_bounds = (1e-2, 9)  # lwc 1e-6, 3e-4
    len_scale_x = np.append(len_scale_x, 0.07)
    len_scale_y = np.append(len_scale_y, 0.07)
    len_scale_t = np.append(len_scale_t, 35)
    len_scale_xbnd = (0.02, 0.5)
    len_scale_ybnd = (0.02, 0.5)
    len_scale_tbnd = (1, 200)
    ind = 25
    for i in range(4):
        if(i>=2):
            ind = 50
            x, ytt = criss_cross(Xt_grid, data_interest, sigma_n, i-2, train_sam_num, mf, ind)
        else:
            x, ytt = criss_cross(Xt_grid, data_interest, sigma_n, i, train_sam_num, mf, ind)
        XT = np.append(XT, x, axis=0)
        yT = np.append(yT, ytt)
        train_sam_num += int(len(x) / 3 + 6)
        num_points = np.append(num_points, train_sam_num - 6 * (i + 1))
        train_per = (train_sam_num - 6 * (i + 1)) / float(4900)
        kx = C(constant_value=sigma_f_x[-1] ** 2,
               constant_value_bounds=(sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
             RBF(length_scale=len_scale_x[-1], length_scale_bounds=len_scale_xbnd)
        kt = C(constant_value=sigma_f[-1] ** 2,
               constant_value_bounds=(sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
             RBF(length_scale=len_scale_t[-1], length_scale_bounds=len_scale_tbnd)
        # kernel = C(constant_value=sigma_f[-1] ** 2, constant_value_bounds = (sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
        #     RBF(length_scale= (len_scale_x[-1],len_scale_y[-1],len_scale_t[-1]), length_scale_bounds = (len_scale_xbnd,len_scale_ybnd,len_scale_tbnd))
        k= kx * kt
        # gpp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=50).fit(XT, yT.reshape(-1, 1))
        gpp = GaussianProcessRegressor(kernel=k, alpha=sigma_n ** 2, n_restarts_optimizer=50).fit(XT,yT.reshape(-1,1))

        # sigma = np.sqrt(gpp.kernel_.k1.get_params()['constant_value'])
        # lx, ly, lt = gpp.kernel_.k2.get_params()['length_scale']

        siga_f_x = np.sqrt(gpp.kernel_.k1.get_params()['k1__constant_value'])
        siga_f = np.sqrt(gpp.kernel_.k2.get_params()['k1__constant_value'])
        lx = gpp.kernel_.k1.get_params()['k2__length_scale']
        lt = gpp.kernel_.k2.get_params()['k2__length_scale']

        sigma_f = np.append(sigma_f, siga_f)
        sigma_f_x = np.append(sigma_f_x, siga_f_x)
        len_scale_x = np.append(len_scale_x, lx)
        #len_scale_y = np.append(len_scale_y, ly)
        len_scale_t = np.append(len_scale_t, lt)

        test = np.concatenate((Xt, np.array([[tr[train_sam_num-1]], ] * Xt.shape[0])), axis=1) # at last observed time
        y_pred, std_pred = gpp.predict(test, return_std=True)
        y_pred_grid = np.reshape(y_pred, (-1, 70)).T
        std_pred = std_pred.reshape(4900, 1)
        y = data_interest[train_sam_num-1, 0]
        yt = y.flatten()
        y_error = yt - y_pred[:, 0]
        y_mse = np.sqrt(np.mean(np.square(y_error)))
        mse = np.append(mse, y_mse)
        lml = gpp.log_marginal_likelihood()
        lmlt = np.append(lmlt, lml)


        plt.figure(figsize=(24, 6), dpi=62)
        plt.subplot(1, 4, 1, xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title('GT at %2dth second (1 Point taken per s)' % (tr[train_sam_num-1]), size=14)
        plt.scatter(XT[:, 0], XT[:, 1], c="black", marker="x", s=8)
        plt.imshow(y.T, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest', norm=norm_cloud)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.subplot(1, 4, 2, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title(
            "Prediction $lx$: %dm, $lt$: %4.2fs, $\sigma_f$: %3.2fm/s" % (lx * 1000, lt, sigma),
            size=14)
            #.set_title("Prediction $lx$: %dm, $ly$: %dm, $lt$: %4.2fs, $\sigma_f$: %3.2fm/s" % (lx * 1000, ly*1000, lt, sigma), size=14)
        plt.imshow(y_pred_grid, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest',norm=norm_cloud)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069), visible=False)
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.subplot(1, 4, 3, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title("Predicted Standard Deviation", size=14)
        plt.imshow(np.reshape(std_pred, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=std_pred.min(), vmax=std_pred.max(), cmap="jet")
        plt.clim(0, 2.25)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069), visible=False)
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.subplot(1, 4, 4, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title("Error, RMSE: %3.2fm/s" % (y_mse), size=14)
        plt.imshow(np.reshape(y_error, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=y_error.min(),
                   vmax=y_error.max(), cmap="jet")
        plt.clim(-2, 5)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069), visible=False)
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.tight_layout()
        plt.show()
        if (save_path):
            plt.savefig(save_path + str(train_sam_num) + 'points.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale_x[1:] * 1000)
    plt.scatter(num_points, len_scale_x[1:] * 1000, c="black", marker="x")
    plt.text(num_points[0], len_scale_x[1] * 1000, round(len_scale_x[1] * 1000, 2), transform=plt.gca().transData, fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale_x[-1] * 1000, round(len_scale_x[-1] * 1000, 2), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$lx$ (m)", fontsize=18)
    plt.title("Evolution of x lengthscale  (Init: $lx$= %dm, Bounds=(%dm, %dm))" % (
    len_scale_x[0] * 1000, len_scale_xbnd[0] * 1000, len_scale_xbnd[1] * 1000), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'xlengthscale.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale_y[1:] * 1000)
    plt.scatter(num_points, len_scale_y[1:] * 1000, c="black", marker="x")
    plt.text(num_points[0], len_scale_y[1] * 1000, round(len_scale_y[1] * 1000, 2), transform=plt.gca().transData, fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale_y[-1] * 1000, round(len_scale_y[-1] * 1000, 2), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$ly$ (m)", fontsize=18)
    plt.title("Evolution of spatial lengthscale  (Init: $ly$= %dm, Bounds=(%dm, %dm))" % (
    len_scale_y[0] * 1000, len_scale_ybnd[0] * 1000, len_scale_ybnd[1] * 1000), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'ylengthscale.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale_t[1:])
    plt.scatter(num_points, len_scale_t[1:], c="black", marker="x")
    plt.text(num_points[0], len_scale_t[1], "%d"%(len_scale_t[1]), transform=plt.gca().transData, fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale_t[-1], "%d"%(len_scale_t[-1]), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$lt$ (s)", fontsize=18)
    plt.title("Evolution of Time lengthscale (Init: $lt$= %ds, Bounds=(%ds, %ds))" % (
    len_scale_t[0], len_scale_tbnd[0], len_scale_tbnd[1]), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lengthscale_t.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.scatter(num_points, sigma_f[1:], c="black", marker="x")
    plt.plot(num_points, sigma_f[1:])
    plt.text(num_points[0], sigma_f[1], round(sigma_f[1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], sigma_f[-1], round(sigma_f[-1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$\sigma_f $(unit: m/s)", fontsize=16)
    plt.title("Evolution of Signal Standard Deviation (Init: $\sigma_f$: %3.2f, Bounds=(%3.2f, %3.2f))" % (
    sigma_f[0], sigma_f_bounds[0], sigma_f_bounds[1]), fontsize=18)
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
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("LML Score", fontsize=18)
    plt.title("Evolution of Log Marginal Likelihood Score", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lml.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, mse)
    plt.scatter(num_points, mse, c="black", marker="x")
    plt.text(num_points[0], mse[0], round(mse[0], 2), transform=plt.gca().transData, fontsize=16, ha='center', va='bottom',
             color='red', weight='bold')
    plt.text(num_points[-1], mse[-1], round(mse[-1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("RMSE Score (unit: m/s)", fontsize=18)
    plt.title("Evolution of Root Mean Square Error", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'rmse.png')

def run_gp3D2(save_path, data_interest):

    # * GP on 3d data with optimization on 2d data
    # * Fly 3 drones on simple line trajectory

    train_sam_num = 0
    XT = np.empty((0, 3), float)
    yT = []
    lmlt = []
    sigma_f = []
    mse = []
    num_points = []
    len_scale_x = []
    len_scale_y = []
    len_scale_t = []
    sigma_n = 0.2  # lwc 1e-6
    sigma_f = np.append(sigma_f, 12)  # lwc 4 #
    sigma_f_bounds = (1e-2, 9)  # lwc 1e-6, 3e-4
    len_scale_x = np.append(len_scale_x, 0.07)
    len_scale_t = np.append(len_scale_t, 35)
    len_scale_xbnd = (0.02, 0.5)
    len_scale_ybnd = (0.02, 0.5)
    len_scale_tbnd = (1, 200)
    ind = 25
    for i in range(4):
        if(i>=2):
            ind = 50
            x, ytt = criss_cross(Xt_grid, data_interest, sigma_n, i-2, train_sam_num, mf, ind)
        else:
            x, ytt = criss_cross(Xt_grid, data_interest, sigma_n, i, train_sam_num, mf, ind)
        XT = np.append(XT, x, axis=0)
        yT = np.append(yT, ytt)
        train_sam_num += int(len(x)/3 +6)
        num_points = np.append(num_points, train_sam_num - 6*(i+1))
        train_per = (train_sam_num - 6*(i+1))/ float(4900)

        # For optimization, just one time use
        nwx = np.vstack([np.concatenate((0.02*np.arange(len(x)/3), 0.02*np.arange(len(x)/3), 0.02*np.arange(len(x)/3))), x[:,-1]])
        nwx= np.transpose(nwx)
        kernel = C(constant_value=sigma_f[-1] ** 2, constant_value_bounds = (sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
            RBF(length_scale= (len_scale_x[-1],len_scale_t[-1]), length_scale_bounds = (len_scale_xbnd,len_scale_tbnd))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n ** 2, n_restarts_optimizer=50).fit(nwx, ytt.reshape(-1, 1))
        sigma = np.sqrt(gp.kernel_.k1.get_params()['constant_value'])
        ld, lt = gp.kernel_.k2.get_params()['length_scale']
        len_scale_t = np.append(len_scale_t, lt)
        len_scale_x = np.append(len_scale_x, ld)
        sigma_f = np.append(sigma_f, sigma)

        # for prediction (so far gathered data)
        pkernel = C(constant_value=sigma_f[-1] ** 2, constant_value_bounds = (sigma_f_bounds[0] ** 2, sigma_f_bounds[1] ** 2)) * \
            RBF(length_scale= (len_scale_x[-1],len_scale_x[-1],len_scale_t[-1]), length_scale_bounds = (len_scale_xbnd,len_scale_xbnd,len_scale_tbnd))
        gpp = GaussianProcessRegressor(kernel=pkernel, alpha=sigma_n ** 2, n_restarts_optimizer=0,optimizer=None).fit(XT, yT.reshape(-1, 1))
        test = np.concatenate((Xt, np.array([[tr[train_sam_num-1]], ] * Xt.shape[0])), axis=1)
        y_pred, std_pred = gpp.predict(test, return_std=True)
        y_pred_grid = np.reshape(y_pred, (-1, 70)).T
        y_pred_b = y_pred_grid
        std_pred = std_pred.reshape(4900, 1)
        y = data_interest[train_sam_num-1, 0]
        yt = y.flatten()
        y_error = yt - y_pred[:, 0]
        y_mse = np.sqrt(np.mean(np.square(y_error)))
        mse = np.append(mse, y_mse)
        lml = gp.log_marginal_likelihood()
        lmlt = np.append(lmlt, lml)


        plt.figure(figsize=(24, 6), dpi=62)
        plt.subplot(1, 4, 1, xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title('GT at %2dth second (1 Point taken per s)' % (train_sam_num - 6*(i+1)), size=14)
        plt.scatter(XT[:, 0], XT[:, 1], c="black", marker="x", s=8)
        plt.imshow(y.T, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest', norm=norm_cloud)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.subplot(1, 4, 2, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title("Prediction $l$: %dm, $lt$: %4.2fs, $\sigma_f$: %3.2fm/s" % (ld * 1000, lt, sigma), size=14)
        plt.imshow(y_pred_grid, origin='lower', extent=cloud_extent, cmap=cmap_cloud, interpolation='nearest',norm=norm_cloud)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069), visible=False)
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.subplot(1, 4, 3, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title("Predicted Standard Deviation", size=14)
        plt.imshow(np.reshape(std_pred, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=std_pred.min(), vmax=std_pred.max(), cmap="jet")
        plt.clim(0, 2.25)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069), visible=False)
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.subplot(1, 4, 4, aspect='equal', xlabel="x coordinate(km)", ylabel="y coordinate(km)") \
            .set_title("Error, RMSE: %3.2fm/s" % (y_mse), size=14)
        plt.imshow(np.reshape(y_error, (-1, 70)).T, origin='lower', extent=cloud_extent, vmin=y_error.min(),
                   vmax=y_error.max(), cmap="jet")
        plt.clim(-2, 5)
        plt.xticks(np.arange(xr[0], xr[-1] + 0.069, 2 * 0.069))
        plt.yticks(np.arange(yr[0], yr[-1] + 0.069, 2 * 0.069), visible=False)
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)

        plt.tight_layout()
        plt.show()
        if (save_path):
            plt.savefig(save_path + str(train_sam_num) + 'points.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale_x[1:] * 1000)
    plt.scatter(num_points, len_scale_x[1:] * 1000, c="black", marker="x")
    plt.text(num_points[0], len_scale_x[1] * 1000, round(len_scale_x[1] * 1000, 2), transform=plt.gca().transData, fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale_x[-1] * 1000, round(len_scale_x[-1] * 1000, 2), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$ld$ (m)", fontsize=18)
    plt.title("Evolution of spatial lengthscale  (Init: $l$= %dm, Bounds=(%dm, %dm))" % (
    len_scale_x[0] * 1000, len_scale_xbnd[0] * 1000, len_scale_xbnd[1] * 1000), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lengthscale.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, len_scale_t[1:])
    plt.scatter(num_points, len_scale_t[1:], c="black", marker="x")
    plt.text(num_points[0], len_scale_t[1], "%d"%(len_scale_t[1]), transform=plt.gca().transData, fontsize=16,
             ha='center', va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], len_scale_t[-1], "%d"%(len_scale_t[-1]), transform=plt.gca().transData,
             fontsize=16, ha='center', va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$lt$ (s)", fontsize=18)
    plt.title("Evolution of Time lengthscale (Init: $lt$= %ds, Bounds=(%ds, %ds))" % (
    len_scale_t[0], len_scale_tbnd[0], len_scale_tbnd[1]), fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lengthscale_t.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.scatter(num_points, sigma_f[1:], c="black", marker="x")
    plt.plot(num_points, sigma_f[1:])
    plt.text(num_points[0], sigma_f[1], round(sigma_f[1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='red', weight='bold')
    plt.text(num_points[-1], sigma_f[-1], round(sigma_f[-1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("$\sigma_f $(unit: m/s)", fontsize=16)
    plt.title("Evolution of Signal Standard Deviation (Init: $\sigma_f$: %3.2f, Bounds=(%3.2f, %3.2f))" % (
    sigma_f[0], sigma_f_bounds[0], sigma_f_bounds[1]), fontsize=18)
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
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("LML Score", fontsize=18)
    plt.title("Evolution of Log Marginal Likelihood Score", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'lml.png')

    plt.figure(figsize=(13, 6), dpi=70)
    plt.plot(num_points, mse)
    plt.scatter(num_points, mse, c="black", marker="x")
    plt.text(num_points[0], mse[0], round(mse[0], 2), transform=plt.gca().transData, fontsize=16, ha='center', va='bottom',
             color='red', weight='bold')
    plt.text(num_points[-1], mse[-1], round(mse[-1], 2), transform=plt.gca().transData, fontsize=16, ha='center',
             va='bottom', color='green', weight='bold')
    plt.xticks(num_points)
    plt.xlabel("Number of data points acquired (1 point/second)", fontsize=18)
    plt.ylabel("RMSE Score (unit: m/s)", fontsize=18)
    plt.title("Evolution of Root Mean Square Error", fontsize=18)
    if (save_path):
        plt.savefig(save_path + 'rmse.png')

"""
#If loading data from scratch
# path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
# mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
#           for minute in range(1, 16) #orig 60
#           for second in range(1, 61)]
# atm = MesoNHAtmosphere(mfiles, 1)
# 
# 
# lwc_data = atm.data['RCT'][349:599,89:90,90:160,170:240]  #89:90 means only 1 height 1.125 km85:123 range of z
# zwind_data = atm.data['WT'][349:599,89:90,90:160,170:240] #85:123 range of z
# 
# ids1,counter1,clouds1=cloud.cloud_segmentation(lwc_data)
# clouds1=list(set(clouds1.values()))
# length_point_clds = np.ndarray((0,1))
# for each_cloud in clouds1:
#     temp = len(each_cloud.points)
#     length_point_clds = np.vstack((length_point_clds,temp))
# 
# sorted_indices = length_point_clds[:,0].argsort()[::-1] # clouds sorted acc to #cloud_points
# cloud1 = clouds1[sorted_indices[0]] #Biggest cloud
# 
# cloud1.calculate_attributes(lwc_data,zwind_data) #zwind also
# #cloud1.calculate_attributes(lwc_data)
# 
# lwc_cloud1 = np.zeros(lwc_data.shape)
# for point in cloud1.points:
#     lwc_cloud1[point] = 1
# del clouds1
# all_Zs=atm.data["VLEV"][:,0,0]
"""

# Dumping to store in pickle
# pickle_out = open("lwc_cloud1.pickle","wb")
# pickle.dump(lwc_cloud1, pickle_out)
# pickle_out.close()
# pickle_out = open("lwc_data.pickle","wb")
# pickle.dump(lwc_data, pickle_out)
# pickle_out.close()
# pickle_out = open("zwind_data.pickle","wb")
# pickle.dump(zwind_data, pickle_out)
# pickle_out.close()
# pickle_out = open("all_Zs.pickle","wb")
# pickle.dump(all_Zs, pickle_out)
# pickle_out.close()

##############
# to load data ! in some env, without latin works=> so remove latin if needed
pickle_in = open("lwc_data.pickle","rb")
lwc_data = pickle.load(pickle_in) #, encoding='latin1')
pickle_in.close()
pickle_in = open("lwc_cloud1.pickle","rb")
lwc_cloud1 = pickle.load(pickle_in) #, encoding='latin1')
pickle_in.close()
pickle_in = open("all_Zs.pickle","rb")
all_Zs = pickle.load(pickle_in) #, encoding='latin1')
pickle_in.close()
pickle_in = open("zwind_data.pickle","rb")
zwind_data = pickle.load(pickle_in) #, encoding='latin1')
pickle_in.close()


xr =np.arange(0.005 + 90*0.01, 0.005 + 160*0.01,0.01)
yr= np.arange(0.005 + 170*0.01, 0.005 + 240*0.01,0.01)
zr = all_Zs[89:90]
tr = np.arange(349,599)

Xt_grid = np.array(np.meshgrid(xr,yr)).T
Xt = Xt_grid.reshape(-1,2)

cloud_extent = [xr[0], xr[-1], yr[0], yr[-1]]

cmap_cloud, norm_cloud = get_color_nd_norm("zwind")
mf = np.array([]) # zero mean prior
save_path = ""

run_gp3D(save_path, zwind_data) #method 1
#run_gp3D2(save_path, zwind_data) #method 2
