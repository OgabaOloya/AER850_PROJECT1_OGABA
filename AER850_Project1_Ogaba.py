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

