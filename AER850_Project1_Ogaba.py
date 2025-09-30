# ================================
#Ogaba Oloya
#AER 850 Project 1
#501097689

#------------------------------------------------------------------------------
#Step 1: Data Processing 

#Importing our relevant toolkits into the system for usage later
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Reading the data from the file 
data = pd.read_csv("AER850_Project1Data/Project 1 Data.csv")

#Checking to verify that all of the columns are valid and to further assign manually by reading without headers
if any (col.lower().startswith("unnamed")for col in data.columns):
    data = pd.read_csv(
        "AER850_Project1Data/Project 1 Data.csv", 
        header=None, 
        names=["X", "Y", "Z", "Step"])

#Displaying the first few rows as a quick verification check 
print(data.head())


#------------------------------------------------------------------------------
#Step 2: Data Visualization 
print("\n\n-----------------Step 2: Data Visualization-----------------\n\n")

#Setting up boxplot subplots for X, Y, and Z
plt.figure(figsize=(15,5))

#Box plot for the variable X
plt.subplot(1,3,1)
data.boxplot(column='X',by='Step',grid=False)
plt.title('Distribution of X per Step')

#Removing of the automatic title which will be applied to Y and Z plots
plt.suptitle('')
plt.xlabel('Step')
plt.ylabel('X')

#Box plot for the variable Y
plt.subplot(1,3,2)
data.boxplot(column='Y',by='Step',grid=False)
plt.title('Distribution of Y per Step')
plt.suptitle('')
plt.xlabel('Step')
plt.ylabel('Y')

#Box plot for the variable Z
plt.subplot(1,3,3)
data.boxplot(column='Z',by='Step',grid=False)
plt.title('Distribution of Z per Step')
plt.suptitle('')
plt.xlabel('Step')
plt.ylabel('Z')

plt.tight_layout()
plt.show()
