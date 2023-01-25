# This project is the data analysis portion of the microrheology summer project.
# Here, we read the data from Excel files and perform analytical techniques and/or
# graph the results using those statistical techniques.
# Shri Deshmukh August-September 2019

# import inspect                        # To be used for looking into package source code
import data_algorithm as alg  
import pandas as pd 
import numpy as np
import math
from scipy import constants
from scipy.special import gamma
import matplotlib.pyplot as plt


def fitToPowerLaw(series):
    '''This function takes in a Pandas Series and calculates a power law with respect to that series.
    In context of this project, the expected series is an emsd that is not logarithmized already.
    This function uses linear regression to determine the power law.'''
    arrTime = list(series.index.values)
    arrMSD = list(series.values)

    if (len(arrMSD) != len(arrTime)):
        raise ValueError('Input arrays must be of the same length.')
    
    arr1 = []        # MSD Array
    arr2 = []        # Time Array
    # Updating our arrays with respect to the data given
    for i in range(len(arrMSD)):
        if (arrMSD[i] == 0.0):
            arr1.append(0.0)
        if (arrMSD[i] == 0.0):
            arr2.append(0.0)
        if (arrMSD[i] != 0.0 or arrTime[i] != 0.0):
            arr1.append(math.log(arrMSD[i]))
            arr2.append(math.log(arrTime[i]))


    # Create a m x 2 matrix entirely of ones, then add the log time
    npArr = np.ones((len(arrTime), 2))
    for i in range(len(arrTime)):
        npArr[i, 1] = arr2[i]  

    # Matrix equation for the linear regression
    pseudoinverse = np.linalg.pinv(np.dot(npArr.T, npArr))
    intermediate = np.dot(pseudoinverse, npArr.T)
    cVector = np.dot(intermediate, arr1)
    
    valueA = math.exp(cVector[0])
    valueN = cVector[1]

    return (valueA, valueN)

def findViscoElasticModulus(emsd, binSize=1, graphAlpha=False):
    ''' This function takes in a Pandas series, which represents the ensemble MSD and calculates G(s)
    from it. Instead of using the direct Laplace transforms, which can cause large errors near frequency 
    extremes (due to truncation of data set), this function uses the algebraic estimation of these transforms
    as established by Thomas Mason (2000). Returns a DataFrames with indexes frequencies and columns
    representing G(s), G'(s), and G''(s)'''   
    values = np.log10(emsd.to_numpy(copy=True))
    times = np.copy(emsd.index.values)
    logtimes = np.log10(times)
    countLim = len(values)
    
    derivatives = []
    frequencies = []
    # Calculating Alpha, Pandas Series holding logarithmic slope of MSD at t=1/s
    for i in range(countLim):
        finiteDiff = 0
        if (i == 0):
            # Forward Finite Difference from i=1 to i=2
            finiteDiff = (values[2] - values[1]) / (logtimes[2] - logtimes[1])
        if (i == countLim - 1):
            # Backwards Difference Formula
            finiteDiff = (values[i] - values[i-1]) / (logtimes[i] - logtimes[i-1])
        else:
            # Central Difference Formula
            finiteDiff = (values[i+1] - values[i-1]) / (logtimes[i+1] - logtimes[i-1])
        derivatives.append(finiteDiff)
        frequencies.append(1 / times[i])    
    

    # Calculating Rolling Average of Alpha
    alpha = pd.Series(data=derivatives, index=frequencies, copy=True)
    movingAvg = alpha.rolling(binSize, min_periods=1, center=True).mean()
    avgDerivs = movingAvg.values
    viscoValues = []
    temperature = 293   # 20 degrees Celsius to Kelvin
    radius = 0.5        # Particle radius in micrometers
    warning = False     # Indicates if any G(s) is negative.

    if (graphAlpha):
        fig, ax = plt.subplots()
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        # ax.set_ylim([-2,2])
        ax.plot(emsd.index, avgDerivs, 'o')
        ax.plot(emsd.index, 0.00001 * emsd.index, '-')
        plt.show()


    # 1e18 changes units to Pascals, Calculating G(s) values
    for i in range(countLim):
        gVal = (constants.Boltzmann * 1e18) * temperature / radius / constants.pi 
        gVal = gVal / emsd[times[i]] 
        gVal = gVal / gamma(1 + avgDerivs[i])
        viscoValues.append(gVal)
        if (gVal < 0):
            warning = True

    if (warning):
        print('\n NOTICE: Some values in G(s) are negative. This may affect overall G\' and G\'\' values.\n') 
    
    # Calculating G' and G''
    gPrime = []
    gPrime2 = []
    for i in range(countLim):
        innerVal = constants.pi * avgDerivs[i] / 2
        gPrime.append(viscoValues[i] * np.cos(innerVal))
        gPrime2.append(viscoValues[i] * np.sin(innerVal))
    
    df_return = pd.DataFrame({
        'G(s)': viscoValues,
        'G\'(s)': gPrime,
        'G\'\'(s)': gPrime2},
        index=frequencies)
    return df_return
    
fileLocation = 'C:\\Shri\\Princeton_2019_Research\\0826_Microrheology_15\\0826_TPX2_MSD.xlsx'

# imsd_read = pd.read_excel(fileLocation, sheet_name='Run 001 IMSD', index_col= 0)
df_emsd = pd.read_excel(fileLocation, sheet_name='Run 001 EMSD', index_col= 0)
# Needed for making emsd_read into a Pandas Series
emsd_read = df_emsd.loc[:,0]

# alg.calculateLinError(imsd_read, emsd_read)   
# alg.calculateLogError(imsd_read, emsd_read, numErrorBars=5000)
# alg.calculateLogError(imsd_read, emsd_read, numErrorBars=10)

df = findViscoElasticModulus(emsd_read, binSize=3, graphAlpha=True)

print(df)
gee = df.loc[:, "G(s)"]
geePrime = df.loc[:, "G'(s)"]
geeDoublePrime = df.loc[:, "G''(s)"]

# Graphing G, G', G''
fig, ax = plt.subplots()
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.plot(gee.index, gee, 'o')
plt.show()

fig, ax = plt.subplots()
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.plot(geePrime.index, geePrime, 'o')
plt.show()

fig, ax = plt.subplots()
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.plot(geeDoublePrime.index, geeDoublePrime, 'o')
plt.show()