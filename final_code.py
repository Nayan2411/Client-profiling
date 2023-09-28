#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import plotly.express as px
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sav = pd.read_csv("downloads/user_profiling_project/sav.csv", sep=",")
product = pd.read_csv("downloads/user_profiling_project/product.csv", sep=",")
shipped_orders = pd.read_csv("downloads/user_profiling_project/shipped_orders.csv", sep=",")
print("Number of datapoints:", len(sav))
print("Number of datapoints:", len(product))
print("Number of datapoints:", len(shipped_orders))


# In[4]:


sav.info()
product.info()
shipped_orders.info()


# In[5]:


sav.head()


# In[6]:


df1_merged = sav.merge(product, on = 'product_id', how='left')
df1_merged.info()


# In[7]:


df1_merged['merchant_id'].nunique()


# In[8]:


df1_merged['client_id'].nunique()


# In[9]:


df1_merged['orderline_id'].nunique()


# In[10]:


df1_merged['SAV_id'].nunique()


# In[11]:


df1_merged[['merchant_id','SAV_id','orderline_id','client_id','product_id']] = df1_merged[['merchant_id','SAV_id','orderline_id','client_id','product_id']].astype(str)


# In[12]:


shipped_orders['merchant_id'].nunique()


# In[13]:


df1_merged.info()


# In[14]:


df1_merged.head()


# In[15]:


print(df1_merged.columns.tolist())


# In[16]:


df1_merged['avg_rating'] = df1_merged[['avg_rate_speed','avg_rate_kindness','avg_rate_relevance']].mean(axis=1)


# In[17]:


df1_merged


# In[19]:


# starts 4-5: Positive(1), stars 1-2: Negative(3), stars 3: Neutral(2) 
def map_sentiment(rating):
    if(float(rating)==3):
        return 2
    elif(float(rating) in range(1,2)):
        return 3
    elif(float(rating)==0):
        return 4
    else:
        return 1;


# In[20]:


review_sentiments=[map_sentiment(s) for s in df1_merged['avg_rating']]
df1_merged['sentiments'] = review_sentiments


# In[25]:


# starts 4-5: Positive(1), stars 1-2: Negative(3), stars 3: Neutral(2) 
def map_sentiments(sen):
    if(int(sen)==3):
        return "Negative"
    elif(int(sen)==2):
        return "Neutral"
    elif(int(sen)==1):
        return "Positive"
    else:
        return "no_rating";


# In[26]:


sentiment_name=[map_sentiments(s) for s in df1_merged['sentiments']]
df1_merged['sentiment_name'] = sentiment_name


# In[30]:


fig = px.pie(df1_merged, values='merchant_id', names='sentiment_name',color='sentiment_name',color_discrete_map={'Neutral':'yellow','Negative':'cyan','no_rating':'red','Positive':'green'})
fig.update_layout(
     autosize=False,
     title='Distribution of Review Sentiments'
    )
fig.show()


# In[31]:


# stars 4-5: Positive(4-5), stars 1-2: Negative(1-2), stars 3: Neutral(3) 
def idv_sentiments(idv):
    if(int(idv) == 1):
        return "Negative"
    elif(int(idv) == 2):
        return "Negative"
    elif(int(idv) == 3):
        return "Neutral"
    elif(int(idv) == 4):
        return "Positive"
    elif(int(idv) == 5):
        return "Positive"
    else:
        return "no_rating";


# In[32]:


sentiment_speed=[idv_sentiments(i) for i in df1_merged['avg_rate_speed']]
df1_merged['sentiment_speed'] = sentiment_speed


# In[34]:


fig = px.pie(df1_merged, values='merchant_id', names='sentiment_speed',color='sentiment_speed',color_discrete_map={'Neutral':'yellow','Negative':'cyan','no_rating':'red','Positive':'green'})
fig.update_layout(
     autosize=False,
     title='Distribution of Speed Sentiments'
    )
fig.show()


# In[37]:


sentiment_kindness=[idv_sentiments(d) for d in df1_merged['avg_rate_kindness']]
df1_merged['sentiment_kindness'] = sentiment_kindness


# In[38]:


fig = px.pie(df1_merged, values='merchant_id', names='sentiment_kindness',color='sentiment_kindness',color_discrete_map={'Neutral':'yellow','Negative':'cyan','no_rating':'red','Positive':'green'})
fig.update_layout(
     autosize=False,
     title='Distribution of Kindness Sentiments'
    )
fig.show()


# In[39]:


sentiment_relevance=[idv_sentiments(v) for v in df1_merged['avg_rate_relevance']]
df1_merged['sentiment_relevance'] = sentiment_relevance


# In[40]:


fig = px.pie(df1_merged, values='merchant_id', names='sentiment_relevance',color='sentiment_relevance',color_discrete_map={'Neutral':'yellow','Negative':'cyan','no_rating':'red','Positive':'green'})
fig.update_layout(
     autosize=False,
     title='Distribution of Relevance Sentiments'
    )
fig.show()


# In[36]:


df1_merged.info()


# In[49]:


from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)


# In[42]:


df1_merged.info()


# In[43]:


da = df1_merged.copy()
cols_del = ['avg_rate_speed', 'avg_rate_kindness', 'avg_rate_relevance','orderline_id','label_product', 'creation_date','last_message_date','sentiments','avg_rating']
df_3 = da.drop(cols_del, axis=1)


# In[44]:


df_3.head()


# In[45]:


df_3[['label']] = df_3[['label']].fillna("unknown")
df_3[['brand ']] = df_3[['brand ']].fillna("unknown")


# In[46]:


df_3.info()


# In[50]:


#Get list of categorical variables
s = (df_3.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[51]:


#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    df_3[i]=df_3[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")


# In[52]:


#Scaling
scaler = StandardScaler()
scaler.fit(df_3)
scaled_df3 = pd.DataFrame(scaler.transform(df_3),columns= df_3.columns )
print("All features are now scaled")


# In[53]:


scaled_df3.head()


# In[54]:


#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_df3)
PCA_df3 = pd.DataFrame(pca.transform(scaled_df3), columns=(["col1","col2", "col3"]))
PCA_df3.describe().T


# In[55]:


#A 3D Projection Of Data In The Reduced Dimension
x =PCA_df3["col1"]
y =PCA_df3["col2"]
z =PCA_df3["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()


# In[56]:


# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_df3)
Elbow_M.show()


# In[ ]:


#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=5)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df3)
PCA_df3["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
df_3["Clusters"]= yhat_AC


# In[ ]:


#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_df3["Clusters"], marker='o', cmap = cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()


# In[ ]:




