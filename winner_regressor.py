import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np 
#import matplotlib.pyplot as plt
#import seaborn as sbn
#import tensorflow as tf
import pickle
#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

#Importing the dataset
"""Dataset is taken from a website by webscraping for independent varibles 
   and for dependent variable we have used a third party API and then I 
   have combined the data. The whole data is from the year 2013 to 2018
   in the city of Bhopal meadured for every day in a year"""

#Column Description
d_Col_Desc={
            "T":"Average temperature",
            "TM":"Maximum temperature",
            "Tm":"Minimum temperature",
            "SLP":"Atmospheric pressure at sea level",
            "H":"Average relative humidity",
             "W":"Average visibility",
            "V":"Average wind speed",
            "VM":"Maximum sustained wind speed",
            "PM 2.5":"Fine particulate matter",
}
df=pd.read_csv("DATA.csv")
df.dropna(inplace=True)
df.replace(to_replace="-",value=np.nan,inplace=True)
df.dropna(inplace=True)
df.set_index(pd.Index(range(1087)),inplace=True)
l=list(df.columns[:-1])
for i in l:
   for j in df[i]:
      df.replace(to_replace=j,value=float(j),inplace=True)

#Creating the dependent and independent variable
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,ra"ndom_state=0)
"""ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="linear"))

#Compiling the model
ann.compile(optimizer="adam",loss="mean_squared_error",metrics=["mean_squared_error"])


#Training the model
ann.fit(X,y,batch_size=32,epochs=100)# 32 is also the dafault value for batch_size

#Predicting the PM 2.5
#X_test is the input given by the user
                    
pickle.dump(ann,open("model.pkl","wb"))"""


n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features=["auto","sqrt"]
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]

random_parameters={
                  "n_estimators":n_estimators,
                  "max_features":max_features,
                  "max_depth":max_depth,
                  "min_samples_split":min_samples_split,
                  "min_samples_leaf":min_samples_leaf,
}
from sklearn.ensemble import RandomForestRegressor
rforest=RandomForestRegressor()
rforest_regressor=RandomizedSearchCV(estimator=rforest,param_distributions=random_parameters,scoring="neg_mean_squared_error",n_iter=100,cv=5,verbose=2,random_state=42)
rforest_regressor.fit(X,y)


pickle.dump(rforest_regressor,open("model.pkl","wb"))
#y_ann_pred=y_ann_pred.flatten()
#array.flatten() will convert 2d array to 1d array

model=pickle.load(open("model.pkl","rb"))
                    