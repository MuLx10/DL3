
# coding: utf-8

# ### Importing libraries

# In[10]:


import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from glob import glob
#get_ipython().magic(u'matplotlib inline')


# ### Data Exploration

# In[11]:


# Reading the train and test meta-data files
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[12]:


train.head()


# In[15]:


# Shape of training and test datasets
print ('Training dataset consists of {} images with {} attributes'.format(train.shape[0], train.shape[1]-1))
# Shape of training and test datasets
print ('Testing dataset consists of {} images.'.format(test.shape[0]))


# Let's have a look at the columns of the training data.

# In[16]:


print('Columns in the dataset:\n\n', train.columns)


# ###  Data Visualization
# 
# Now we will visualize our data to get a better understanding of it.

# We will begin with visualizing the distribution of the labels in the training data.

# In[21]:


cols = list(train.columns)
cols.remove('Image_name')
cols.sort()


# In[22]:


count_labels = train[cols].sum()


# In[24]:


count_labels.sort_values(inplace=True)


# From the figure below, we can see that there are 85 different attributes/ labels and Attrib_21 is common in almost all animals while Attrib_66 is rare.

# In[25]:

'''
plt.figure(figsize=(18, 8))
ax = sns.barplot(x=count_labels.index, y=count_labels.values)
ax.set_xticklabels(labels=count_labels.index,rotation=90, ha='right')
ax.set_ylabel('Count')
ax.set_xlabel('Attributes/ Labels')
ax.title.set_text('Label/ Attribute distribution')
plt.tight_layout()
'''

# In[26]:


label_data = np.array(train[cols])


# Next we will compute the co-occurrence matrix for the labels. 

# In[27]:


# Compute the cooccurrence matrix
cooccurrence_matrix = np.dot(label_data.transpose(), label_data)
print('\n Co-occurence matrix: \n', cooccurrence_matrix)


# In[28]:


# Compute the cooccurrence matrix in percentage
# Refrence: https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas/20574460
cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
with np.errstate(divide = 'ignore', invalid='ignore'):
    cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal))


# In[29]:


print('\n Co-occurrence matrix paercentage: \n', cooccurrence_matrix_percentage)


# From the plot of the co-occurence matrix (below), we can see which labels(or attributes) genreally occur together.

# In[30]:

'''
ax = plt.figure(figsize=(18, 12))

sns.set(style='white')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio 
sns.heatmap(cooccurrence_matrix_percentage, cmap=cmap, center=0, square=True, linewidths=0.15, cbar_kws={"shrink": 0.5})

plt.title('Co-occurrence Matrix of the Labels')
'''

# We define the paths to the image folders.

# In[31]:


TRAIN_PATH = 'data/train/'
TEST_PATH = 'data/test/'


# In[32]:


img_path = TRAIN_PATH+str(train.Image_name[0])


# We import the OpenCV and Python Image library for image manipulation. 

# In[33]:


from PIL import Image
import cv2


# In[34]:


Image.open(img_path)


# The computer cannot see shapes or colors. It reads each image as an array of numbers.

# In[35]:


img = cv2.imread(img_path)
img


# In[36]:


# Shape of each image
img.shape


# In[38]:


# Extracting label columns
label_cols = list(set(train.columns) - set(['Image_name']))
label_cols.sort()


# In[39]:


# Extracting labels corresponding to image at the zeroth index of the training dataset.
labels = train.iloc[0][2:].index[train.iloc[0][2:] == 1]


# We plot the Animal and the attributes/ labels corresponding to it.

# In[43]:

'''
txt = 'Labels/ Attributes: ' + str(labels.values)
ax = plt.figure(figsize=(10, 10))
ax.text(.5, .05, txt, ha='center')
plt.imshow(img)
'''

# In the image above we can see a Rhinoceros and all the attributes associated with him.

# ### Data Preprocessing
# 
# Next, we will preprocess our image data before supplying it to the training model.

# In[44]:


from tqdm import tqdm
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    return img


# The above function reads an image and resizes it to 128 x 128 dimensions and returns it.

# In[28]:


#temp = train.sample(frac=0.3)


# In[29]:


#train = temp.reset_index(drop=True)


# In[30]:


train_img = []

for img_path in tqdm(train.Image_name.values):
    train_img.append(read_img(TRAIN_PATH + img_path))


# In[31]:


import gc


# In[32]:


# Convert the image data into an array. 
# Since the range of color(RGB) is in the range of (0-255).
# Hence by dividing each image by 255, we convert the range to (0.0 - 1.0)

X_train = np.array(train_img, np.float32) / 255.


# In[33]:


del train_img
gc.collect()


# Next, we will calculate the mean and standard deviation.

# In[34]:


mean_img = X_train.mean(axis=0)


# In[35]:


std_dev = X_train.std(axis = 0)


# Next, we will normalize the image data using the following formula: 
# 
# <center>** X = (x - mean of x)/(std. deviation of x)**<center/>
# 

# In[36]:


X_norm = (X_train - mean_img)/ std_dev


# In[37]:


X_norm.shape


# In[38]:


del X_train


# In[39]:


gc.collect()


# In[40]:


y = train[label_cols].values


# In[41]:


from sklearn.model_selection import train_test_split


# Finally, we create the training and validation sets.

# In[42]:


Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_norm, y, test_size=0.05, random_state=47)


# In[43]:


del X_norm
gc.collect()


# ### Model Architecture
# 
# We will be using the Keras framework to create our model. But you may also use other frameworks like Tensorflow, Pytorch, etc.

# In[44]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

print(K.image_data_format(),Xtrain.shape)

# For this problem, we create a model from scratch. We will use a Sequential model, which is a linear stack of layers to build this model.

# In[45]:
input_shape = Xtrain.shape[1:]

gc.collect()


# In[46]:


model = Sequential()
model.add(BatchNormalization(input_shape=Xtrain.shape[1:]))
#model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(85, activation='sigmoid'))


# Here we generate the summary our model. We can see that there are approximately 10 million parameters to train. 

# In[47]:


model.summary()


# Next, we define our loss function, the optimizer and metrics for our model.

# In[48]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[49]:


early_stops = EarlyStopping(patience=3, monitor='val_acc')
checkpointer = ModelCheckpoint(filepath='weights.best.eda.hdf5', verbose=1, save_best_only=True)


# Finally, we train our model.

# In[50]:


model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=47, batch_size=32, callbacks=[checkpointer], verbose=1)


# In[52]:


train_pred = model.predict(Xtrain).round()


# In[53]:


from sklearn.metrics import f1_score
f1_score(ytrain, train_pred, average='samples')


# In[54]:


valid_pred = model.predict(Xvalid).round()


# In[56]:


f1_score(yvalid, valid_pred, average='samples')


# In[58]:


#del Xtrain
#del Xvalid
#del ytrain 
#del yvalid
gc.collect()


# ### Prediction on Test Set
# 
# Now that we have built and trained our model, we will use it to predict the labels of the test images.

# In[51]:


from sklearn.metrics import f1_score


# In[59]:


test_img = []
for img_path in tqdm(test.Image_name.values):
    test_img.append(read_img(TEST_PATH + img_path))


# In[60]:


X_test = np.array(test_img, np.float32) / 255.


# In[61]:


del test_img
gc.collect()


# The test images are normalized below.

# In[62]:


mean_img = X_test.mean(axis=0)


# In[63]:


std_dev = X_test.std(axis = 0)


# In[64]:


X_norm_test = (X_test - mean_img)/ std_dev


# In[65]:


del X_test
gc.collect()


# Predict the labels on the test images.

# In[66]:


model.load_weights('weights.best.eda.hdf5')


# In[67]:


pred_test = model.predict(X_norm_test).round()


# In[68]:


pred_test = pred_test.astype(np.int)


# #### Creating the submission file

# In[70]:


subm = pd.DataFrame()


# In[71]:


subm['Image_name'] = test.Image_name


# In[72]:


label_df = pd.DataFrame(data=pred_test, columns=label_cols)


# In[73]:


subm = pd.concat([subm, label_df], axis=1)


# In[74]:


subm.to_csv('submit.csv', index=False)


# END.....
