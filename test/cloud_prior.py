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
from nephelae_mapping.test.util import save_pickle
from nephelae_mapping.test.process_map import border_cs
from nephelae_mapping.test.flight_moving_frame import get_coordinate_extent, show_map
from matplotlib.patches import Ellipse


def fitEllipse(cont,method):

    # Fitting an ellipse to a curve using Direct least squares fitting.
    # Inspired from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    x=cont[:,0]
    y=cont[:,1]
    x=x[:,None]
    y=y[:,None]
    D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S=np.dot(D.T,D)
    C=np.zeros([6,6])
    C[0,2]=C[2,0]=2
    C[1,1]=-1
    E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))
    if method==1:
        n=np.argmax(np.abs(E))
    else:
        n=np.argmax(E)
    a=V[:,n]
    #-------------------Fit ellipse-------------------
    b,c,d,f,g,a=a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
    num=b*b-a*c
    cx=(c*d-b*f)/num
    cy=(a*f-b*d)/num
    angle=0.5*np.arctan(2*b/(a-c))*180/np.pi
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    a=np.sqrt(abs(up/down1))
    b=np.sqrt(abs(up/down2))
    #---------------------Get path---------------------
    ell=Ellipse((cx,cy),a*2.,b*2.,angle)
    ell_coord=ell.get_verts()
    params=[cx,cy,a,b,angle]
    return params,ell_coord

if __name__ == "__main__":

    gt_save = False
    anim_save = False
    save_path = "exp/4/"
    if(anim_save):
        matplotlib.use("Agg")
        plt.rcParams['animation.ffmpeg_path'] = '/home/dlohani/miniconda3/envs/nephelae/bin/ffmpeg'

    atm = MFDataset("/net/skyscanner/volume1/data/Nephelae/MesoNH-2019-02/REFHR.1.ARMCu.4D.nc")
    tvar, zvar, xvar, yvar = get_coordinate_extent(atm)

    var_interest = 'RCT' # or 'WT' etc
    data = MesoNHVariable(atm, var_interest, interpolation='linear')
    lwc_unit = "kg a/kg w"

    t = 211        # inital timestamp
    zStart = 1002  # initial height  ini m
    z_depth = 128  # length of z to travel in cloud
    xSlice = slice(1702, 2172, None)
    ySlice = slice(4170, 4700, None)
    lwc_extent = [data[t,:,xSlice,ySlice].data.min(), data[t,:,xSlice,ySlice].data.max() + 0.25 * data[t,:,xSlice,ySlice].data.max()]

    xyBounds = data[t, zStart, xSlice, ySlice].bounds
    xyExtent = [xyBounds[0][0], xyBounds[0][1], xyBounds[1][0], xyBounds[1][1]]

    data_shape = data[t, zStart, ySlice, xSlice].data.shape
    fig, lwc_data = show_map(np.zeros(data_shape), xyExtent, lwc_unit, lwc_extent, 0, 0)
    _, border_data = border_cs(np.zeros((data_shape[0]*data_shape[1],1)), data_shape, xyExtent, threshold=3e-5, c="Black")
    ellip_curve, = plt.plot(0, 0)
    ellip_center = plt.scatter(0, 0, c="Red")

    def init():
        #do nothing
        pass

    def update(i):

        z = zStart + i
        plt.title("Cloud at t= %ds & z= %dm" % (t, z))
        lwc_data_z = data[t, z, ySlice, xSlice].data
        lwc_data.set_data(lwc_data_z)

        global border_data

        # Remove old border plot
        for coll in border_data.collections:
            plt.gca().collections.remove(coll)

        # Get new border plot and data
        _, border_data = border_cs(lwc_data_z, data_shape, xyExtent, threshold=3e-5, c="Black")

        # Get border curve coords
        curve_coords = border_data._get_allsegs_and_allkinds()[0][0][0]

        # Fit Ellipse on curve coords
        ell_params, ellipse = fitEllipse(curve_coords, 1)
        ellip_curve.set_data(ellipse[:, 0], ellipse[:, 1])
        ellip_center.set_offsets(np.c_[ell_params[0], ell_params[1]])

    anim = animation.FuncAnimation(
        fig,
        update,
        frames = z_depth,
        init_func = init,
        repeat = False,
        interval = 1,
    )

    if (anim_save):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(save_path+'gt_ellipse.mp4', writer=writer)
    else:
        plt.show(block=False)

    #=====================================================================================================

    # chk = data[t, zStart+1, ySlice, xSlice].data
    # area = chk[chk >= 1e-5].shape[0] * 25**2
    #print(len(lwc_z_mean))
    # plt.figure()
    # plt.plot(range(z_depth), lwc_z_mean)
    # m, c = estimate_coef(np.arange(z_depth), lwc_z_mean)
    #plt.plot()

    # lwc_z_mean = []
    # lwc_z_mean.append(lwc_data_z.mean())
