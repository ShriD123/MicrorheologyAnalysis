# This project is the data algorithm portion of the microrheology summer project.
# Here, we input nd2 files to provide the frames for each of the images, and then 
# gather the data from that directly using Trackpy's algorithms. 
# Shri Deshmukh June-September 2019


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import math
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from pims import ND2_Reader
import pims
import trackpy as tp
import matplotlib as mpl
import matplotlib.pyplot as plt
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

import os
# Just to ignore the UserWarning of calling FramesSequenceND.__init__() cuz it's annoying
import warnings
# warnings.filterwarnings('ignore')

# Just to prevent any scope errors
strPath, particleSize, minimumIntensity, maxDisplacement = None, None, None, None
framesMissing, minFrames, micronsPerPixel, framesPerSecond = None, None, None, None
traj, demo, graphs, test = None, None, None, None


# Functions For Declaring Values
def findDataFile(fileLocation):
    '''This function defines the path of the file location.'''
    # Pathname of the file intended for analysis. Remember to use \\ for backslash between folders.
    # Don't include \\ backslash at the very end of the string. For more info, check os.path
    global strPath
    strPath = fileLocation

    assert(os.path.exists(strPath))

def getPathName():
    '''This function returns the path name of the file in strPath.'''
    global strPath
    return strPath

def declareParticleProperties(particle_size, min_intensity, max_displacement):
    '''This function defines important properties of the particle. For more information, check the 
    __main__ portion of the data_analysis file.'''
    global particleSize 
    global minimumIntensity
    global maxDisplacement

    particleSize = particle_size
    minimumIntensity = min_intensity
    maxDisplacement = max_displacement 

    assert(particleSize % 2 == 1)
    assert(maxDisplacement >= particleSize)    

def declareFrameProperties(min_frames, mPP, fPS, frames_missing=0):
    '''This function declares important values for the image frames containing the data. For more 
    information, check the __main__ portion of the data_analysis file.'''
    global framesMissing  
    global minFrames      
    global micronsPerPixel                        
    global framesPerSecond

    framesMissing = frames_missing
    minFrames = min_frames
    micronsPerPixel = mPP
    framesPerSecond = fPS

    assert(isinstance(micronsPerPixel, float))
    assert(isinstance(framesPerSecond, float))

def declareBooleanVariables(showTrajectories, conductDemo, showIMSDGraph, testVariables):
    '''Declare important boolean variables here. This affects the output of the program.'''
    global traj
    global demo
    global graphs 
    global test 

    traj = showTrajectories
    demo = conductDemo
    graphs = showIMSDGraph
    test = testVariables
    
# Functions For Conducting the Data Analysis
def useTrackpyFit(series):
    '''This function takes in a Pandas Series object that represents the ensemble mean-squared 
    displacement over time. It then performs the linear best fit in log space using the trackpy 
    algorithm (which I'm not sure what the algorithm is) and plots.'''
    assert(type(series) is Series)

    plt.figure()
    plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    plt.xlabel('lag time $t$')
    print(tp.utils.fit_powerlaw(series))             # Holds the values of the best fit for the power law At^n 
    tp.utils.fit_powerlaw(series)                    # performs linear best fit in log space, plots

def graphIMSD(imsd, scale='log'):
    '''This function graphs the individual msd, which is inputted to be of the type Pandas Data Frame. 
    Scale indicates the relative scaling of the axes. To conduct linear scaling, use the string "linear".'''
    assert(type(imsd) is DataFrame)

    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.plot(imsd.index, imsd, 'k-', alpha=0.1)  # black lines, semitransparent
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
    plt.show()
    
def graphEMSD(emsd, scale='log'):
    '''This function graphs the ensemble msd, which is inputted to be of type Pandas Series. 
    Scale indicates the relative scaling of the axes. To conduct linear scaling, use the string "linear".'''
    assert(type(emsd) is Series)

    fig, ax = plt.subplots()
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.plot(emsd.index, emsd, 'o')
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
    # ax.plot(emsd.index, 0.00001 * emsd.index, '-')
    plt.show()

def analyzeData(sliceSize=0, max_time=100):
    '''This function performs the Trackpy analysis. Indicate sliceSize (int) for the size of the batch
    you want to use. By default, it is set to 0 (which indicates using the whole file). Max_time represents 
    the maximum lag time up to which the msd is calculated. It's best to set this as the smallest length of all the 
    frames that were imaged for this experiment.
    This function also returns the individual msd calculated by Trackpy as a Pandas DataFrame.'''
    assert(sliceSize >= 0)

    # Here we are reading in the files and creating the frames.
    print('\nMake sure you\'ve changed the variables according to this file! If not, press Ctrl+C to abort. \n')

    print('Accessing the file: ' + strPath)
    frames = ND2_Reader(strPath)

    if (demo):
        plt.imshow(frames[0])                         # Plots a frame at an index (frame number)
        plt.show()  

        f2 = tp.locate(frames[0], particleSize, invert=False, minmass=minimumIntensity)    
        tp.annotate(f2, frames[0])                      # Indicates which particles are being tracked 


    '''Here we are locating the features (particles) in a particular frame.'''

    if (test):
        teststep = len(frames) // 7                      
        for i in range(len(frames) // teststep):
            f1 = tp.locate(frames[(i * teststep)], particleSize, invert=False, minmass=minimumIntensity)
            tp.annotate(f1, frames[(i * teststep)])

            fig, ax = plt.subplots()
            ax.hist(f1['mass'], bins=20)
            # Optionally, label the axes.
            ax.set(xlabel='mass', ylabel='count')  
            plt.show() 

            # Use only for large data sets to ensure. Check trackpy's walkthrough for details
            plt.figure()
            tp.subpx_bias(f1)                 # We want decimal part of x and y pos as evenly distributed
            plt.show()


    '''Locating features in all frames. Can change slice size for testing purposes.'''   
    if (sliceSize == 0 or sliceSize >= len(frames)):
        f = tp.batch(frames[:], particleSize, minmass=minimumIntensity, invert=False)
    else:
        f = tp.batch(frames[:sliceSize], particleSize, minmass=minimumIntensity, invert=False)


    '''Linking features into particle trajectories.'''
    t = tp.link_df(f, maxDisplacement, memory=framesMissing)


    '''Filtering ephemeral trajectories (not useful in the statistics).'''                          
    t1 = tp.filter_stubs(t, minFrames)

    # Compare the number of particles in the unfiltered and filtered data with the following code snippet.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())


    '''Trace the trajectories and correct for the resulting drift.'''
    t1.index.names = ["index"]
    d = tp.compute_drift(t1)                            # Accounts for any average drift
    tm = tp.subtract_drift(t1.copy(), d)                # Data Frame holding trajectories of all particles                        

    if (traj):
        plt.figure()
        tp.plot_traj(t1)            # For plotting the trajectories. Can indicate drift.

        plt.figure()
        ax = tp.plot_traj(tm)        


    '''Analyzing the trajectories. '''                           
    im = tp.imsd(tm, micronsPerPixel, framesPerSecond, max_lagtime=max_time)      # Panda Data Frame holding MSD of each particle at a certain lag time
    # em = tp.emsd(tm, micronsPerPixel, framesPerSecond)      # Panda Series holding (weighted average) MSD of each particle at a certain lag time

    if (graphs):
        graphIMSD(im)
    
    print('\n')                                     # For the terminal to look nice
    frames.close()                              # Keep after every end of analyzeData()! Very important!

    return im

def analyzeDataFull(sliceSize=0, max_time=100):
    '''This function performs the Trackpy analysis. Indicate sliceSize (int) for the size of the 
    batch you want to use. By default, it is set to 0 (which indicates using the whole file).
    This function also returns the individual msd and ensemble msd calculated by Trackpy as a tuple. Max_time 
    represents the maximum lag time up to which the msd is calculated. Set this as the smallest length of the frames
    in all of the data sets that were taken for this batch. 
    The individual msd is a Pandas DataFrame and ensemble msd is a Pandas Series.'''
    assert(sliceSize >= 0)

    # Here we are reading in the files and creating the frames.
    print('\nMake sure you\'ve changed the variables according to this file! If not, press Ctrl+C to abort. \n')

    print('Accessing the file: ' + strPath)
    frames = ND2_Reader(strPath)

    if (demo):
        plt.imshow(frames[0])                         # Plots a frame at an index (frame number)
        plt.show()  

        f2 = tp.locate(frames[0], particleSize, invert=False, minmass=minimumIntensity)    
        tp.annotate(f2, frames[0])                      # Indicates which particles are being tracked 


    '''Here we are locating the features (particles) in a particular frame.'''

    if (test):
        teststep = len(frames) // 7                      
        for i in range(len(frames) // teststep):
            f1 = tp.locate(frames[(i * teststep)], particleSize, invert=False, minmass=minimumIntensity)
            tp.annotate(f1, frames[(i * teststep)])

            fig, ax = plt.subplots()
            ax.hist(f1['mass'], bins=20)
            # Optionally, label the axes.
            ax.set(xlabel='mass', ylabel='count')  
            plt.show() 

            # Use only for large data sets to ensure. Check trackpy's walkthrough for details
            plt.figure()
            tp.subpx_bias(fl)                 # We want decimal part of x and y pos as evenly distributed
            plt.show()


    '''Locating features in all frames. Can change slice size for testing purposes.'''   
    if (sliceSize == 0 or sliceSize >= len(frames)):
        f = tp.batch(frames[:], particleSize, minmass=minimumIntensity, invert=False)
    else:
        f = tp.batch(frames[:sliceSize], particleSize, minmass=minimumIntensity, invert=False)


    '''Linking features into particle trajectories.'''
    t = tp.link_df(f, maxDisplacement, memory=framesMissing)


    '''Filtering ephemeral trajectories (not useful in the statistics).'''                          
    t1 = tp.filter_stubs(t, minFrames)

    # Compare the number of particles in the unfiltered and filtered data with the following code snippet.
    print('Before:', t['particle'].nunique())
    print('After:', t1['particle'].nunique())


    '''Trace the trajectories and correct for the resulting drift.'''
    t1.index.names = ["index"]
    d = tp.compute_drift(t1)                            # Accounts for any average drift
    tm = tp.subtract_drift(t1.copy(), d)                # Data Frame holding trajectories of all particles                        

    if (traj):
        plt.figure()
        tp.plot_traj(t1)            # For plotting the trajectories. Can indicate drift.

        plt.figure()
        ax = tp.plot_traj(tm)        


    '''Analyzing the trajectories. '''                          
    im = tp.imsd(tm, micronsPerPixel, framesPerSecond, max_lagtime=max_time)      # Panda Data Frame holding MSD of each particle at a certain lag time
    em = tp.emsd(tm, micronsPerPixel, framesPerSecond, max_lagtime=max_time)     # Panda Series holding (weighted average) MSD of each particle at a certain lag time

    if (graphs):
        graphIMSD(im)
        graphEMSD(em)
    
    print('\n')                                     # For the terminal to look nice
    frames.close()                              # Keep after every end of analyzeData()! Very important!

    return im, em

def calculateAverage(imsd):
    '''This function calculates the average ensemble MSD from an inputted DataFrame giving the individual MSD  
    of many particles. This is different from Trackpy's EMSD in that Trackpy uses a weighted average, where
    this uses an arithmetic mean. This function returns the ensemble MSD average over time as a Pandas Series.'''
    series = imsd.mean(axis=1, skipna=True)
    return series

def calculateLinError(imsd, emsd):
    ''' This function calculates the standard error and plots the standard error on a linear scale.'''
    interval = int(len(emsd) // 20)
    stDev = imsd.std(axis=1, skipna=True)
    
    plt.errorbar(emsd.index, emsd, yerr=stDev, errorevery=interval, fmt='o', 
    ecolor='g', capsize=5)
    plt.xlabel('Lag Time')
    plt.ylabel('Mean Squared Displacement')
    plt.yscale('linear', nonposy='clip')
    plt.xscale('linear', nonposx='clip')
    plt.show()

def calculateOldLogError(imsd, emsd, numErrorBars):
    '''This function calculates the standard error and plots the standard error on a log scale.'''
    interval = int(len(emsd) // numErrorBars)
    stDev = imsd.std(axis=1, skipna=True)
     
    plt.errorbar(emsd.index, emsd, yerr=stDev, errorevery=interval, fmt='o', 
    ecolor='g', capsize=5)
    plt.xlabel('Lag Time')
    plt.ylabel('Mean Squared Displacement')
    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposx='clip')
    plt.show()

def calculateLogError(imsd, emsd, numErrorBars):
    ''' This function calculates the relative error (https://faculty.washington.edu/stuve/log_error.pdf)
    and plots the relative error on a log scale. For short timescales, numErrorBars should be high (~100); 
    for long time scales, numErrorBars should be low (~10). numErrorBars must be a postive integer.'''
    assert(numErrorBars > 0 and type(numErrorBars) is int)

    interval = int(len(emsd) // numErrorBars)
    if (interval < 1):
        interval = 1

    stDev = imsd.std(axis=1, skipna=True) 
    indexes = list(imsd.index.values)
    logErr = []

    for i in range(len(emsd)):
        newValue = stDev[indexes[i]] / (10 * emsd[indexes[i]])
        logErr.append(abs(newValue))
    
    plt.errorbar(emsd.index, emsd, yerr=logErr, errorevery=interval, fmt='o', 
    ecolor='g', capsize=5)
    plt.xlabel('Lag Time')
    plt.ylabel('Mean Squared Displacement')
    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposx='clip')
    plt.show()

def calculatePercentError(imsd, emsd):
    '''This function calculates the percent error of the standard deviation with respect to mean and plots
    it over time on a linear scale. '''
    stDev = imsd.std(axis=1, skipna=True)
    indexes = list(imsd.index.values)
    perErr = []
    
    for i in range(len(emsd)):
        error = stDev[indexes[i]] / emsd[indexes[i]] * 100
        perErr.append(error)

    fig, ax = plt.subplots()
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.plot(indexes, perErr, 'o')
    plt.show()
    



if __name__ == '__main__':
    '''Declare important variables here.'''
    # Pathname of the file intended for analysis. Remember to use \\ for backslash between folders.
    # Don't include \\ backslash at the very end of the string. For more info, check os.path
    strPath = 'C:\\Shri\\Princeton 2019 Research\\0705_Microrheology_08\\PEO\\PEO_05_001.nd2'         

    # Particle size changes for each file. To discern a new particle size (in pixels), go through a few frames and 
    # zoom in to several particles to determine a good particle size. The size must be an odd integer,
    # and it is better to err on the large side. For reference, in 0622_Microrheology_05, a particle size
    # of 25 was used (you can use that number to see a relative amount)
    particleSize = 11       

    # Approximated minimum signal intensity of the particles. This can change per file. To determine a good value, 
    # use the code supplied after minimum intensity to make a histogram. Then, choose a minmass value that seems to
    # be a good minimum. Then, annotate a few random frames to ensure no noise is interpreted as a particle. 
    minimumIntensity = 10000 

    # With some testing, it seems a good maximum is equal to about one particle size (occasionally a bit more). 
    # To test, look through consecutive frames and make an educated guess about the maximum displacement in pixels.
    # Note: Adaptive Search not implemented due to potential issues with the data analysis.
    maxDisplacement = 45             

    # Number of frames the program keeps a particle's ID for if it goes missing. Best kept at 0.
    framesMissing = 0      

    # Minimum number of frames the particle must be present for statistics. This references the minimum
    # time it takes a particle to sample a^2 diffusively.
    minFrames = 10        

    # This value can be found by putting the nd2 file in ImageJ. This changes for every file.
    # At the top of the file gives the dimensions of the file in microns & pixels.
    micronsPerPixel = 164.92 / 1280    

    # This value is determined by dividing number of frames by the video time (in seconds).                  
    framesPerSecond = 4.0

    # This value is determined by setting the smallest number of frames in the combined data sets for
    # a particular experiment.
    max_lagtime = 717

    assert(particleSize % 2 == 1)
    assert(maxDisplacement >= particleSize) 
    assert(isinstance(micronsPerPixel, float))
    assert(isinstance(framesPerSecond, float))
    assert(os.path.exists(strPath))  

    declareBooleanVariables(False, False, False, True)

    # The actual meat of the program.
    df_imsd = analyzeData(max_time=max_lagtime)
    df_emsd = calculateAverage(df_imsd)
                 







