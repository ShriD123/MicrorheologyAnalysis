# This project is the data gathering portion of the microrheology summer project.
# Here, we provide the direct user interface to gather adequate data from several files 
# and write them to an Excel document for use in fitting later. 
# Shri Deshmukh June-September 2019

import data_algorithm as alg
import pandas as pd
import os

# Make sure to include the .xlsx at the end of the string (and \\).
excelFileLocation = 'C:\\Shri\\Princeton 2019 Research\\0622_Microrheology_05\\Water\\0622_Water_MSD.xlsx'


# Don't delete this! This implements a user check for you to make sure you don't accidently lose data.
assert(os.path.exists(excelFileLocation))
print('\nALERT! Make sure you are writing to the correct Excel File Location! Any data currently in the Excel File sheet will be overwritten.')
print('The location is ' + excelFileLocation + '.')
str1 = input("Are you sure you want to continue? (Y/N) ")   

if (str1.upper() != 'Y'):
    exit()


alg.declareParticleProperties(25, 100000, 25)   
alg.declareFrameProperties(3, 132.49/1024, 465/120)
alg.declareBooleanVariables(True, False, False, False)   

alg.findDataFile('C:\\Shri\\Princeton 2019 Research\\0622_Microrheology_05\\Water\\run_001.nd2')
imsd1 = alg.analyzeData(sliceSize=0, max_time=463)
alg.findDataFile('C:\\Shri\\Princeton 2019 Research\\0622_Microrheology_05\\Water\\run_002.nd2')
imsd2 = alg.analyzeData(sliceSize=0, max_time=463)
alg.findDataFile('C:\\Shri\\Princeton 2019 Research\\0622_Microrheology_05\\Water\\run_003.nd2')
imsd3 = alg.analyzeData(sliceSize=0, max_time=463)

# Use tolerance less than 1 / framesPerSecond otherwise aggregate data gon be corrupt
df1 = pd.merge_asof(imsd1, imsd2, on='lag time [s]', tolerance=0.030, direction='nearest')
df2 = pd.merge_asof(df1, imsd3, on='lag time [s]', tolerance=0.030, direction='nearest')
df2.set_index('lag time [s]', inplace=True)
df3 = alg.calculateAverage(df2)

# df1.set_index('lag time [s]', inplace=True)
# df2 = alg.calculateAverage(df1)

print('Graphing Aggregates...\n')
alg.graphIMSD(df2, scale='linear')
alg.graphEMSD(df3, scale='linear')
alg.graphIMSD(df2)
alg.graphEMSD(df3)


# Writing to Excel.
with pd.ExcelWriter(excelFileLocation) as writer:
    print('Writing...', end='\r')
    imsd1.to_excel(writer, sheet_name='Run 001 IMSD')
    imsd2.to_excel(writer, sheet_name='Run 002 IMSD')
    imsd3.to_excel(writer, sheet_name='Run 003 IMSD')
    df2.to_excel(writer, sheet_name='Aggregate IMSD')
    df3.to_excel(writer, sheet_name='Aggregate EMSD')
    print('Writing...Done!')


'''
# Merging the data and graphing. This is the full_outer_join method
# Merge_asof with a tolerance might be better for time series
df1 = pd.merge(imsd1, imsd2, on='lag time [s]', how='outer')
df_outer_imsd = pd.merge(df1, imsd3, on='lag time [s]', how='outer')
'''

