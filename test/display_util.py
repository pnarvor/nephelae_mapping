################################################################
############ Utility functions to display cloud in 2D/3D #######
################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

time_index = 0

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