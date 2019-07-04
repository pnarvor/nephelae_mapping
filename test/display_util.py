################################################################
############ Utility functions to display cloud in 2D/3D #######
################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import pickle
from mesonh_atm.mesonh_atmosphere import MesoNHAtmosphere

time_index = 0

def get_color_nd_norm(type="lwc", c=[36, 5, 100, 17], b=[-10e-5,35e-5,1e-5]):
    #TODO: need to update / not a good way
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

def hori_cloud(data, ti, cmap, norm, cloud_extent, tr, zr, data_name, zi=0):
    # Displays cloud horizontal cross section at fixed height and time
    # Inputs:
    #   data - zwind or lwc grid : shape (nx X ny)
    #   ti - time index
    #   zi - height index
    # Outputs: 2D slice with title and units

    plt.imshow(data.T, origin='lower', extent=cloud_extent, cmap=cmap, interpolation='nearest', norm=norm)
    plt.title("Time: %2dth sec, Height: %dm"% (tr[ti], zr[zi]*1000))
    plt.xlabel("x coordinate(km)")
    plt.ylabel("y coordinate(km)")
    plt.xticks(np.arange(cloud_extent[0], cloud_extent[1] + 0.069, 2 * 0.069))
    plt.yticks(np.arange(cloud_extent[2], cloud_extent[3] + 0.069, 2 * 0.069))
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    if(data_name=="lwc"):
        cbar = plt.colorbar(fraction=0.0471, pad=0.01)
        cbar.ax.set_title('kg w/kg a', fontsize=15, pad=14)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    elif(data_name=="zwind"):
        cbar = plt.colorbar(fraction=0.0471, pad=0.01, format='%.1f')
        cbar.ax.set_title('   m/s', fontsize=15, pad=7)
    else:
        cbar = plt.colorbar(fraction=0.0471, pad=0.01)

def dynamic_hori_slice(data_interest, cmap, norm, cloud_extent, tr, zr, data_name="zwind"):
    # Displays cloud horizontal cross section at fixed height
    # and varying time with left/right key press
    # Inputs:
    #   data_interest - zwind or lwc data as 4D numpy array
    #   cmap - color map to use
    #   norm - normalization of color for imshow
    #   cloud_extent - [min(x), max(x), min(y), max(y)] in kms for plot
    #   tr - time range
    #   zr - height range
    # Outputs: matplotlib at initial time -> press left/right arrows to navigate

    def press(event):
        # Handles key press event
        global time_index
        if event.key == 'right':
            time_index = time_index + 1
        elif event.key == 'left':
            time_index = time_index - 1
        y = data_interest[time_index, 0]
        plt.clf()
        hori_cloud(y, time_index, cmap, norm, cloud_extent, tr, zr, data_name)
        plt.draw()

    fig = plt.figure()
    y = data_interest[0, 0]
    fig.canvas.mpl_connect('key_press_event', press)
    hori_cloud(y, 0, cmap, norm, cloud_extent, tr, zr, data_name)
    plt.draw()
    global time_index
    time_index = 0

if __name__ == "__main__":

    ### If loading data from scratch
    # path = "/net/skyscanner/volume1/data/mesoNH/ARM_OneHour3600files_No_Horizontal_Wind/"
    # mfiles = [path+"U0K10.1.min{:02d}.{:03d}_diaKCL.nc".format(minute, second)
    #           for minute in range(1, 16) #orig 60
    #           for second in range(1, 61)]
    # atm = MesoNHAtmosphere(mfiles, 1)
    #
    #lwc_data = atm.data['RCT'][349:449, 89:139, 90:160, 170:240]
    #zwind_data = atm.data['WT'][349:449, 89:139, 90:160, 170:240]
    #all_Zs = atm.data["VLEV"][:, 0, 0]

    # Saving Data
    # pickle_out = open("lwc_data_with_z.pickle", "wb")
    # pickle.dump(lwc_data, pickle_out)
    # pickle_out.close()
    # pickle_out = open("zwind_data_with_z.pickle", "wb")
    # pickle.dump(zwind_data, pickle_out)
    # pickle_out.close()
    # pickle_out = open("all_Zs.pickle","wb")
    # pickle.dump(all_Zs, pickle_out)
    # pickle_out.close()

    ##############
    # Load data ! In some env, without latin works=> so remove latin if needed
    pickle_in = open("lwc_data_with_z.pickle", "rb")
    lwc_data = pickle.load(pickle_in)  # , encoding='latin1')
    pickle_in.close()
    pickle_in = open("all_Zs.pickle", "rb")
    all_Zs = pickle.load(pickle_in)  # , encoding='latin1')
    pickle_in.close()
    pickle_in = open("zwind_data_with_z.pickle", "rb")
    zwind_data = pickle.load(pickle_in)  # , encoding='latin1')
    pickle_in.close()

    xr = np.arange(0.005 + 90 * 0.01, 0.005 + 160 * 0.01, 0.01)
    yr = np.arange(0.005 + 170 * 0.01, 0.005 + 240 * 0.01, 0.01)
    zr = all_Zs[89:139] # fetch height array
    tr = np.arange(349, 449) # fetch time array
    cloud_extent = [xr[0], xr[-1], yr[0], yr[-1]]

    #### Example 1 Zwind ######
    cmap_cloud, norm_cloud = get_color_nd_norm("zwind")
    dynamic_hori_slice(zwind_data, cmap_cloud, norm_cloud, cloud_extent, tr, zr, "zwind")

    #### Example 2 lwc   ######
    # cmap_cloud, norm_cloud = get_color_nd_norm("lwc")
    # dynamic_hori_slice(lwc_data, cmap_cloud, norm_cloud, cloud_extent, tr, zr, "lwc")