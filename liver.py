#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scikitplot as skplt
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




# In[ ]:


df = pd.read_csv("indian_liver_patient.csv")




# In[ ]:


df["Albumin_and_Globulin_Ratio"].fillna("0.6", inplace = True) 


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


sns.countplot(data=df, x = 'Dataset', label='Count')
plt.show()

LD, NLD = df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[ ]:





# Data Set Information:
# 
# This data set contains 416 liver patient records and 167 non liver patient records.The data set was collected from north east of Andhra Pradesh, India. Selector is a class label used to divide into groups(liver patient or not). This data set contains 441 male patient records and 142 female patient records. 
# 
# Any patient whose age exceeded 89 is listed as being of age "90".
# 
# 

# Attribute Information:
# 
# 1. Age	Age of the patient 
# 2. Gender	Gender of the patient 
# 3. TB	Total Bilirubin 
# 4. DB	Direct Bilirubin 
# 5. Alkphos Alkaline Phosphotase 
# 6. Sgpt Alamine Aminotransferase 
# 7. Sgot Aspartate Aminotransferase 
# 8. TP	Total Protiens 
# 9. ALB	Albumin 
# 10. A/G Ratio	Albumin and Globulin Ratio 
# 11. Selector field used to split the data into two sets (labeled by the experts) 
# 
# 

# In[ ]:


df_sex = pd.get_dummies(df['Gender'])
df_new = pd.concat([df, df_sex], axis=1)
Droop_gender = df_new.drop(labels=['Gender' ],axis=1 )
Droop_gender.columns = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','Male','Fmale','Dataset']

X = Droop_gender.drop('Dataset',axis=1)
y = Droop_gender['Dataset']


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compile ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the data
history =classifier.fit(X_train, y_train, batch_size = 20, epochs = 50)


# In[ ]:



plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.style.use("ggplot")
plt.figure()
N = 50
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

# In[ ]:



y_pred = classifier.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()