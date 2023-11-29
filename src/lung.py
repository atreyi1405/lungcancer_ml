import pandas as pd
import numpy as num
import seaborn as sns
import matplotlib.pyplot as plt
import dataset
import csv
from sklearn import preprocessing

#cancerset=open("src/dataset/data.csv") 
#print(cancerset.read())
df=pd.read_csv("src/dataset/data.csv")
print(df.duplicated().sum()) #found: no duplicates
df.info() #found patientID, level are object datatypes
df.isnull().sum() #checking for null values

descriptive_stats=df.describe()
print(descriptive_stats)

le=preprocessing.LabelEncoder() 
df['Gender']=le.fit_transform(df['Gender']) #female=0, male=1
df['Level']=le.fit_transform(df['Level']) #convert to numerical value, high=0, low=1, medium=2
#sns.countplot(x='Gender', data=df) 
#plt.title('Target Distribution')
#plt.show() #gender distribution, 
#plt.figure(figsize=(20, 20))

df_new = df.drop(columns=['index', 'Patient Id']) #irrelevant data
cn=df_new.corr()
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.figure(figsize=(16, 20))  # Adjust the figure size to fit within the window

# Adjusting the heatmap, annot_fontsize controls the font size of annotations
sns.heatmap(cn, cmap=cmap, annot=True, square=True, annot_kws={"fontsize": 8})  # Try different font sizes

plt.subplots_adjust(bottom=0.5, top=0.9)
#plt.tight_layout()  # Adjust subplot parameters to fit the plot contents within the window
plt.show()


