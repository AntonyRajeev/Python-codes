
# coding: utf-8

# In[2]:


#importing dependancies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#using pandas to read
data = pd.read_csv('mnist_train.csv')


# In[ ]:


#viewing
data.head()


# In[ ]:


#extracting data
a = data.iloc[3,1:],values


# In[ ]:


#reshaping
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[ ]:


#preparing and seperating
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


# In[ ]:


#creating test and train
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


# In[ ]:


#check data
y_train.head()


# In[ ]:


#callin classifier
rf=RandomClassifier(n_estimators=100)


# In[ ]:


#fit the model
rf.fit(x_train, y_train)


# In[ ]:


#prediction on test data
pred= rf.predict(x_test)


# In[ ]:


pred


# In[ ]:


#check prediction accuracy
a=y_test.values

# calculate number of correctly predicted values
count = 0
for i is range(length(pred)):
    if pred[i]==a[i]
         count=count+1


# In[ ]:


count


# In[ ]:


#total values that the predction code was run on
len(pred)

