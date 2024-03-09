#!/usr/bin/env python
# coding: utf-8

# In[1]:


#For Installing Kaggle
get_ipython().system('pip install kaggle')


# In[3]:


#Upload your Kaggle.json File
#Configuring the path of kaggle.json file
#This is fix Code ti import yout API 
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[6]:


#API to fetech the dataset from kaggle
#Importing twitter sentiment dataset
get_ipython().system('kaggle datasets download -d kazanova/sentiment140')


# In[7]:


#extracting the Compessed Dataset
from zipfile import ZipFile
#importing the zipline from ZipFile
dataset = 'sentiment140.zip'


# In[8]:


with ZipFile(dataset,'r') as zip:
    # r for Read, using as zip for extract
    zip.extractall()
    print ("SucessFul")
    
#Rename the dataset file


# In[9]:


#Importing the dependencies
import numpy as np
import pandas as pd #Data into Structure
import re #Regular Expression
from nltk.corpus import stopwords #Natural Language Tool Kit
from nltk.stem.porter import PorterStemmer #Stemming is used to change the words in root words like Chaning Walks ,Walking to Walk
from sklearn.feature_extraction.text import TfidfVectorizer #Change the words into Numerical Data for Processing of data
from sklearn.model_selection import train_test_split #For Spliting the data into Train and Test
from sklearn.linear_model import LogisticRegression #Basic Ml used in this
from sklearn.metrics import accuracy_score #Accuracy and Performance of the System


# In[10]:


#This are the Words doesn't change(Or add) the Meaning to the Sentence
import nltk
nltk.download('stopwords')
print(stopwords.words('english'))


# In[11]:


#Data Processing
#Loading the data from csv file to pandas Dataframe
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')


# In[12]:


#Checking the Number of Rows and Columns
twitter_data.shape
twitter_data.head() #Printing thr First five Columns


# In[13]:


#Naming the columns and reading the dataset again
column_names = ['target','id','data','flag','user','text']
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv',names=column_names, encoding='ISO-8859-1')


# In[14]:


twitter_data.head() #Printing thr First five Columns


# In[15]:


#Counting the number of missing value in the dataset 
twitter_data.isnull().sum()


# In[16]:


#Checking the distribution of target Columns
twitter_data['target'].value_counts()
#This is done to check the equal distribution of the Positive and Negative Tweets
#If it is not even divided we have to upsampling and Downsampling


# In[17]:


#Convert the target from 4 to 1 this done for making it simple and easy and to Look good
twitter_data.replace({'target':{4:1}},inplace=True)
#Checking the distribution of target Columns
twitter_data['target'].value_counts()
#0 --> Negative 
#1 --> Positive 


# In[18]:


#Stemming :- Stemming is the Process of reducing a word to its Root Words Like Actor, Actress, Acting = Act
port_stem = PorterStemmer()
def stemming(content):
    #Content is the import for the function
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    # Removing all the Charter from the tweet except the A-Z and a-z
    #It remove all the Number, Punchtation, Arrow, Comma, Special char and @, etc
    stemmed_content = stemmed_content.lower()
    #Changing the words to lower as it donesn't the meaning from upper to lower
    stemmed_content = stemmed_content.split()
    # Split thw words and adding into the List
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    #Changing the Word to root word
    #Operation of change the Word into stemmed words which are not Present in the Stopwords
    stemmed_content = ' '.join(stemmed_content)
    #Again joining the Words from List to tweet
    
    return stemmed_content


# In[19]:


twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)


# In[20]:


twitter_data.head()


# In[21]:


print(twitter_data['stemmed_content'])
print(twitter_data['target'])


# In[22]:


#Steparting the data and label
X = twitter_data['stemmed_content'].values
#Storing thr value of text into x
Y = twitter_data['target'].values
#Storing thr value of target into Y


# In[23]:


print(X)


# In[24]:


print(Y)


# In[25]:


#Spliting the data to Training data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)
#test_size = 0.2 means that 20% of the data is test data
#stratify Mean equal distribution of Positive tweet and Negative Tweet
#Random_State will insure that all the people have the same Set of test and Train Because it is always Random selected for Test and Train


# In[26]:


print(X.shape,X_train.shape,X_test.shape)


# In[27]:


print(X_train)


# In[28]:


print(Y_train)


# In[29]:


#Converting the textual data to Numerical Data
#In convert all the text into Numerical
vectorizer = TfidfVectorizer()
#Depending upon the Number of repeat of the words in tweet. 
#Depend upon that word on that what effecting it is making on Positive or Negative Tweet
#All the words are converted into some important Values
X_train = vectorizer.fit_transform(X_train)
#For train we use the fit_transform to transform data in Numerical
X_test = vectorizer.transform(X_test)
#Based upon the training data we transform the test data into numerical data


# In[30]:


print(X_train)


# In[31]:


print(Y_train)


# In[32]:


#Training the ML Model
#Logistic Regression is used as we have just two value input and result alongwith it, input is present in Numerical form
model = LogisticRegression(max_iter=1000)
# Max Itersation is max number of time it can go it is upto 1000


# In[33]:


model.fit(X_train, Y_train)
#It will train the model


# In[34]:


#Model Evalution
#Accuracy Score

#Accuracy score on the training data
#It is True value on which the Model is train
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print("Accuracy score on the training data : ",training_data_accuracy)


# In[35]:


#Accuracy score on the test data
#It is new Value to the Model is tested
X_test_prediction = model.predict(X_test )
test_data_accuracy = accuracy_score(Y_test,X_test_prediction)
print("Accuracy score on the test data : ",test_data_accuracy)
#The accuracy of training data and Test Data Must be equal otherwise it is assumed that the Model is Overfitted


# In[36]:


#To save the Model to use it later, As we have not to train the Model again and again for the best result.
import pickle
filename = 'trained_model.sav' #Name to the file as be different but for this case is trained_model.sav
pickle.dump(model,open(filename,'wb'))
# model is the name of the model we created in the time of Logistic Regression
#wb is write the file in Binary format
#dump is used to create the file 


# In[37]:


#Using the saved Model for future prediction
#Loading the saved Model
loaded_model = pickle.load(open('trained_model.sav','rb'))
#rb mean that reading the file in binary format


# In[38]:


#Testing our model for save model
X_new = X_test[200]
print(Y_test[200])
prediction = model.predict(X_new)
print(prediction)


# In[ ]:




