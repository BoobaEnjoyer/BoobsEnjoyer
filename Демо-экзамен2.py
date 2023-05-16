#!/usr/bin/env python
# coding: utf-8

#Парсинг и предобработка данных

# In[1]:


#Импортирование необходимых библиотек
import pandas as pd
import numpy as np
from pprint import pprint
import codecs
import json
import glob
pd.set_option('display.max_columns', None)
from pandas import json_normalize


# In[3]:


#Путь к файлам .geojson
path = 'Desktop/Data'
file = glob.glob(path + "/*.json")
df_full=pd.DataFrame()

#df_full=pd.DataFrame(columns=['Пост', 'день публикации', 'месяц публикации', 'время публикации'])
#Цикл для получения файла и его загрузки, используя json.load 
for filename in file:
    name = filename.split("\\")[-1][:-5]
    with codecs.open(filename, 'r', 'utf-8-sig') as json_file:  
        data = json.load(json_file)
    
    for article in data['refs']:  
        if article != None:
            df=pd.concat(
                [
                    pd.DataFrame([article[0]],columns=['Post']),
                    json_normalize(article[1]),
                    pd.DataFrame([name],columns=['Company'])
                ],
                axis=1
            )
            df_full=pd.concat([df_full,df],axis=0,ignore_index=True)


# In[4]:


df_full.info()


# In[5]:


df_full.head()


# In[7]:


#Путь к файлам .geojson
path = 'Desktop/Data'
file = glob.glob(path + "/*.json")


df = pd.DataFrame(columns=['rate','subs','industries','about','Company']) 

#Датафрейм с информацией о компании

#Цикл для получения файла и его загрузки, используя json.load 
for filename in file:
    with codecs.open(filename, 'r', 'utf-8-sig') as json_file:  
        data = json.load(json_file)
        name = filename.split("\\")[-1][:-5]   
        try:
            company_info=pd.concat([json_normalize(data['info']),pd.DataFrame([name],columns=['Company'])],axis=1)
        except:
            d={'rate':['Не указано'],'subs':['Не указано'],'industries':['Не указано'],'about':['Не указано']}
            company_info=pd.concat([pd.DataFrame(d),pd.DataFrame([name],columns=['Company'])],axis=1)
    df = pd.concat([df,company_info], axis=0, ignore_index=True)
df.head()    
   


# In[8]:


tk = df_full.merge(df, on='Company',how='left')


# In[9]:


tk.shape


# In[10]:


tk.head()


#Обработка текста

# In[11]:


import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[12]:


sw = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()

def clear_text(text):
    text=text.lower()
    text = re.sub(r'[^а-яё ]','', str(text))
    tokens=word_tokenize(text, language="russian")
    tokens = [morph.parse(i)[0].normal_form for i in tokens]
    tokens = [ i for i in tokens if i not in sw and len(i) > 3]
    return tokens


# In[13]:


tk['lemmatize_tokens'] = tk['Post'].apply(clear_text)


# In[14]:


tk.head()


# In[15]:


tk['clear_text'] = tk['lemmatize_tokens'].apply(lambda x: " ".join(x))


# In[16]:


tk.head()


# In[17]:


tk.to_csv('data.csv', index=False)


#Векторизация текста и поиск ngram


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[19]:


tfidf = TfidfVectorizer(min_df=5,max_df=0.8, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(tk['clear_text'])
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns = tfidf.get_feature_names())
df_tfidf.head()


# In[20]:


X_tfidf


# In[21]:


df_tfidf["Company"]=tk["Company"]


#Кластеризация


# In[22]:


from sklearn.cluster import KMeans, Birch, MiniBatchKMeans
from sklearn.decomposition import PCA


# In[23]:


model = KMeans(n_clusters=5)


# In[24]:


reduced_data = PCA(n_components=2).fit_transform(X_tfidf.toarray())
model.fit_transform(reduced_data)
df_tfidf["cluster"] = model.predict(reduced_data)


# In[25]:


df_tfidf["cluster"]


# In[26]:


from sklearn.metrics import silhouette_score


# In[27]:


print("silhouette_score -", silhouette_score(reduced_data, df_tfidf["cluster"]))


#Классификация


# In[29]:


df=pd.read_json("Desktop/Data/Target.json")
df = df.rename(columns = {"Сompany":"Company"})
df


# In[30]:


df_tfidf["Company"]


# In[31]:


df_tfidf=df_tfidf.merge(df, on='Company')
df_tfidf['Nominations']


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,4))
sns.histplot(data=df_tfidf,x='Nominations')


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x=df_tfidf.drop(['Nominations', "Company"], axis=1)
y=df_tfidf['Nominations']


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)


# In[36]:


from sklearn.tree import DecisionTreeClassifier as Tree


# In[37]:


tree = Tree(max_depth=20, min_samples_split=4, min_samples_leaf=2)


# In[38]:


tree.fit(x_train, y_train)


#Оценка модели


# In[39]:


predictions = tree.predict(x_test)


# In[40]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


# In[ ]:




