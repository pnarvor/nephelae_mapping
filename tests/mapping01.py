#! /usr/bin/python3

import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import animation
import time

# from nephelae_mapping.gprmapping import GprPredictor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as gpk


# noiseStd = 1.0
noiseStd = 0.1
# a        = 10.0
a        = 2.0
f0       = 3.0
T        = 1000
# N        = 10
N        = 20

def process(x):
    return a*np.sin(2.0*np.pi*f0*x)

locations        = np.linspace(0,1.0,T)
trueValues       = process(locations)
noise            = noiseStd*np.random.randn(len(trueValues))
observableValues = trueValues + noise

kernel = (a * gpk.RBF(length_scale= 0.25 / f0)) + gpk.WhiteKernel(noiseStd**2)

def do_update():

    # update values
    indexes          = np.random.randint(0,T-1, N)
    obsLocations     = np.array([locations[i]        for i in indexes])
    obsValues        = np.array([observableValues[i] for i in indexes])
    gprProcessor = GaussianProcessRegressor(kernel,
                                            alpha=0.0,
                                            optimizer=None,
                                            copy_X_train=False)
    gprProcessor.fit(obsLocations.reshape(-1,1), obsValues.reshape(-1,1))
    prediction = gprProcessor.predict(locations.reshape(-1,1), return_std=True)

    # update plot
    axes.cla()
    axes.fill_between(locations,
                      prediction[0].reshape(-1) - 3*prediction[1],
                      prediction[0].reshape(-1) + 3*prediction[1],
                      color=[1.0,0.9,0.9])
    axes.fill_between(locations,
                      prediction[0].reshape(-1) - prediction[1],
                      prediction[0].reshape(-1) + prediction[1],
                      color=[0.9,0.7,0.7])
    axes.plot(locations, observableValues,      label="Observable values")
    axes.plot(locations,       trueValues,      label="True values")
    axes.plot(locations,    prediction[0],      label="Prediction")
    axes.plot(obsLocations,     obsValues, "o", label="Obs values")
    axes.grid()
    axes.legend(loc='upper right')
    axes.set_xlim([0,1.0])
    axes.set_ylim([-2.5*a,2.5*a])


draw = True
fig, axes = plt.subplots(1,1)
def init():
    do_update()
def update(i):
    if draw:
        do_update()
    # time.sleep(0.5)

anim = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval = 1)

plt.show(block=False)






