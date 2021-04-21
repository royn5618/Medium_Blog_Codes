#!/usr/bin/env python
# coding: utf-8

# # Getting Started

# In[2]:


# Upload Kaggle json

get_ipython().system('pip install -q kaggle')
get_ipython().system('pip install -q kaggle-cli')
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp "C:/Users/nroy0/Documents/MyGitHub/Kaggle/kaggle.json" ~/.kaggle/')
get_ipython().system('cat ~/.kaggle/kaggle.json ')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')
get_ipython().system('kaggle competitions download -c fake-news -p dataset')
# !kaggle datasets download -c someone/some-data -p dataset


# In[2]:


get_ipython().system('unzip /content/dataset/train.csv.zip')


# In[3]:


get_ipython().system('unzip /content/dataset/test.csv.zip')


# # Imports

# In[4]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['figure.figsize'] = [10, 10]
import seaborn as sns
sns.set_theme(style="darkgrid")

from wordcloud import WordCloud
from PIL import Image
import PIL.ImageOps
from wordcloud import ImageColorGenerator

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import word_tokenize
import contractions

import time

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding


# In[5]:


train_df = pd.read_csv('/content/train.csv', header=0)
train_df.head()


# In[6]:


train_df.shape


# In[7]:


train_df.info()


# In[8]:


sns.countplot(x='label', data=train_df, palette='Set3')


# In[9]:


test_df = pd.read_csv('/content/test.csv', header=0)
test_df.head()


# In[10]:


test_df.shape


# In[11]:


test_df.info()


# ## Drop all instances which have atleast one column missing

# In[12]:


train_df.dropna(axis=0, how='any',inplace=True)
test_df.dropna(axis=0, how='any',inplace=True)


# In[13]:


train_df.shape, test_df.shape


# In[14]:


train_df.info()


# In[15]:


test_df.info()


# During prediction time title, author and text, i.e. all the current input features will be present.

# ## Check length of Text

# In[16]:


train_df['raw_text_length'] = train_df['text'].apply(lambda x: len(x))


# In[17]:


train_df.head()


# In[18]:


sns.boxplot(y='raw_text_length', data=train_df, palette="Set3")


# In[19]:


sns.boxplot(y='raw_text_length', x='label', data=train_df, palette="Set3")


# In[20]:


len(train_df['author'].unique())


# In[21]:


gen_news_authors = set(list(train_df[train_df['label']==0]['author'].unique()))
fake_news_authors = set(list(train_df[train_df['label']==1]['author'].unique()))


# In[22]:


overlapped_authors = gen_news_authors.intersection(fake_news_authors)


# In[23]:


len(gen_news_authors), len(fake_news_authors), len(overlapped_authors)


# ## Text Cleaning
# 
# 1. Remove special characters
# 2. Expand contractions
# 3. Convert to lower-case
# 4. Word Tokenize
# 5. Remove Stopwords

# In[28]:


def preprocess_text(x):
    cleaned_text = re.sub(r'[^a-zA-Z\d\s\']+', '', x)
    word_list = []
    for each_word in cleaned_text.split(' '):
        try:
            word_list.append(contractions.fix(each_word).lower())
        except:
            print(x)
    return " ".join(word_list)


# **Got Error because of some sort of Turkish/Slavic language**
# 
# ABÇin ilişkilerinde ABD ve NATOnun etkisi yazan Manlio Dinucci Uluslararası bir forumda konuşan İtalyan coğrafyacı Manlio Dinucci ABDnin tüm dünyaya egemen olabilmek için sahip olduğu silahların analizini bireşimleştirdi Suriye Rusya ve Çinin bugün elde silah herkesin açıkça kabul ettiği bu üstünlüğü dünyanın bu tek kutuplu örgütlenişi tartışılır hale getirmesinden dolayı bu makale daha da önem kazanmaktadır
# 
# Accordingly the preprocessing of the texts were executed.

# In[29]:


text_cols = ['text', 'title', 'author']


# In[30]:


get_ipython().run_cell_magic('time', '', 'for col in text_cols:\n    print("Processing column: {}".format(col))\n    train_df[col] = train_df[col].apply(lambda x: preprocess_text(x))\n    test_df[col] = test_df[col].apply(lambda x: preprocess_text(x))')


# In[31]:


get_ipython().run_cell_magic('time', '', 'for col in text_cols:\n    print("Processing column: {}".format(col))\n    train_df[col] = train_df[col].apply(word_tokenize)\n    test_df[col] = test_df[col].apply(word_tokenize)')


# In[32]:


get_ipython().run_cell_magic('time', '', 'for col in text_cols:\n    print("Processing column: {}".format(col))\n    train_df[col] = train_df[col].apply(\n        lambda x: [each_word for each_word in x if each_word not in stopwords])\n    test_df[col] = test_df[col].apply(\n        lambda x: [each_word for each_word in x if each_word not in stopwords])')


# In[33]:


train_df.head()


# # Text Data Exploration

# ## World Cloud Generation

# In[ ]:





# In[ ]:





# # TF-IDF Transformation

# In[40]:


train_df.head()


# In[41]:


train_df['text_joined'] = train_df['text'].apply(lambda x: " ".join(x))
test_df['text_joined'] = test_df['text'].apply(lambda x: " ".join(x))


# In[42]:


train_df.head()


# In[43]:


tf_idf_transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))

# fit train data to the count vectorizer
count_vect_train = count_vectorizer.fit_transform(train_df['text_joined'].values)

#fit the ngrams count to the tfidf transformers
tf_idf_train = tf_idf_transformer.fit_transform(count_vect_train)


# In[44]:


count_vect_train


# In[45]:


tf_idf_train


# In[46]:


target = train_df['label'].values


# # Train Test Split
# 

# In[47]:


X_train, X_test, y_train, y_test = train_test_split(tf_idf_train, target, random_state=0)


# In[48]:


df_perf_metrics = pd.DataFrame(columns  =['Model', 'Accuracy_Training_Set', 'Accuracy_Test_Set', 'Precision', 'Recall', 'f1_score'])


# # Logistic Regression

# In[49]:


log_reg_model = LogisticRegression(C=1e5)
log_reg_model.fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test)


# In[50]:


df_perf_metrics.loc[0] = [
    "Logistic Regression",
    log_reg_model.score(X_train, y_train),
    log_reg_model.score(X_test, y_test),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]


# In[51]:


df_perf_metrics


# In[52]:


df_perf_metrics = pd.DataFrame(columns=[
    'Model', 'Accuracy_Training_Set', 'Accuracy_Test_Set', 'Precision',
    'Recall', 'f1_score', 'Training Time (secs'
])


def get_perf_metrics(model, i):
    model_name = type(model).__name__
    start_time = time.time()
    print("Training {} model...".format(model_name))
    model.fit(X_train, y_train)
    print("Completed {} model training.".format(model_name))
    elapsed_time = time.time() - start_time
    print("Time elapsed: {:.2f} s.".format(elapsed_time))
    y_pred = model.predict(X_test)
    df_perf_metrics.loc[i] = [
        model_name,
        model.score(X_train, y_train),
        model.score(X_test, y_test),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred), "{:.2f}".format(elapsed_time)
    ]
    print("Completed {} model's performance assessment.".format(model_name))


# In[53]:


models_list = [LogisticRegression(C=1e5),
               RandomForestClassifier(n_estimators=5),
               ExtraTreesClassifier(n_estimators=5,n_jobs=4),
               AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)]


# In[54]:


get_ipython().run_cell_magic('time', '', 'for n, model in enumerate(models_list):\n    get_perf_metrics(model, n)')


# In[55]:


df_perf_metrics


# # Adding Title and Author Information to the Text

# In[56]:


train_df['all_info'] = train_df['text'] + train_df['title'] + train_df['author'] 
train_df['all_info'] = train_df['all_info'].apply(lambda x: " ".join(x))

test_df['all_info'] = test_df['text'] + test_df['title'] + test_df['author'] 
test_df['all_info'] = test_df['all_info'].apply(lambda x: " ".join(x))


# In[57]:


tf_idf_transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))

# fit train data to the count vectorizer
count_vect_train = count_vectorizer.fit_transform(train_df['all_info'].values)

#fit the ngrams count to the tfidf transformers
tf_idf_train = tf_idf_transformer.fit_transform(count_vect_train)

X_train, X_test, y_train, y_test = train_test_split(tf_idf_train, target, random_state=0)


# In[58]:


get_ipython().run_cell_magic('time', '', 'for n, model in enumerate(models_list):\n    get_perf_metrics(model, n)')


# In[59]:


df_perf_metrics


# # NLP using Deep Learning with Keras

# In[60]:


all_text = train_df["all_info"].astype(str).tolist()
all_text[0]


# In[61]:


tokenizer = Tokenizer(oov_token = "<OOV>", num_words=6000)
tokenizer.fit_on_texts(all_text)
word_index = tokenizer.word_index
len(word_index)


# In[62]:


sequences = tokenizer.texts_to_sequences(all_text)
padded = pad_sequences(sequences, padding = 'post', maxlen=6000)
padded[0]


# In[63]:


padded.shape


# In[110]:


def get_model():
    model=Sequential()
    model.add(Embedding(6000, 300 ,input_length=6000))
    model.add(Dropout(0.3))
    model.add(LSTM(200))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    return model

model = get_model()
model.summary()


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(padded, target, test_size=0.2)


# In[112]:


callbacks=[
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, 
                                  verbose=1, mode="min", restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath="best_model.hdf5", verbose=1, save_best_only=True)
]


# In[114]:


model = get_model()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[116]:


get_ipython().run_cell_magic('time', '', 'history = model.fit([X_train], \n                    y_train, \n                    epochs=10,\n                    batch_size=64, \n                    validation_data=([X_test], y_test), \n                    callbacks=callbacks)')


# In[117]:


metric_toplot = "loss"
plt.plot(history.epoch, history.history[metric_toplot], ".:", label="loss")
plt.plot(history.epoch,
         history.history["val_" + metric_toplot],
         ".:",
         label="val_loss")
plt.legend()


# In[100]:


model = keras.models.load_model('best_model.hdf5')


# In[118]:


y_pred = model.predict_classes(X_test)


# In[119]:


y_test.shape, y_pred.shape


# In[120]:


precision_score(y_test, y_pred)


# In[121]:


confusion_matrix(y_test, y_pred)


# In[122]:


recall_score(y_test, y_pred)


# In[ ]:





# In[ ]:




