# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
df.head(5)
X = df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']]
y=df['list_price']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=6)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols = X_train.columns
print(cols)

#plt.scatter(x=X_train[col],y=y_train,ax=axes[i*3+j])
fig ,axes = plt.subplots(3,3, figsize=(20,10))
col = cols[0]
# print(type(axes))
# print(np.shape(axes))

# print(axes)
# print(axes[0][0])
# print(axes(0)(1))
for i in range(3):
    for j in range(3):
        col = cols[i*3+j]
        axes[i,j].scatter(x=X_train[col],y=y_train) 
#plt.scatter(x=X_train[col],y=y_train,ax=axes[])     
# code ends here



# --------------
# Code starts here
corr = X_train.corr()
print(corr)
print(corr[abs(corr)>0.75])
X_train.drop(['play_star_rating','val_star_rating'],axis=1,inplace=True)

X_test.drop(['play_star_rating','val_star_rating'],axis=1,inplace=True)


# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
r2=r2_score(y_test,y_pred)
print("Value of MSE is :", mse)
print("Value of r2 is :", r2)
# Code ends here


# --------------
# Code starts here

residual = y_test - y_pred
plt.hist(residual)


# Code ends here


