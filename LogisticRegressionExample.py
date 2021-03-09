import numpy as nm 
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix  ,accuracy_score
from sklearn.linear_model import LogisticRegression  

#importing datasets  
data_set= pd.read_csv('User_Data.csv')  
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)  

cm= confusion_matrix(y_test,y_pred)  
accuracy = accuracy_score(y_test,y_pred)
print(cm,accuracy)