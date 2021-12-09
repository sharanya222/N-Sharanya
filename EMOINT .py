#!/usr/bin/env python
# coding: utf-8

# # Emotion intensity in text

# Importing libraries for data preprocessing

# In[111]:


import pandas as pd
import numpy as np
import string
import re


# Importing libraries required for visualisation

# In[112]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# Importing libraries for NLP

# In[113]:


from collections import Counter
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Packages for modelling

# In[114]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import metrics


# In[115]:


import scipy


# Columns names

# In[116]:


cols = ['id', 'text', 'label', 'intensity']


# Loading data (text files)

# In[117]:


anger_train = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/anger_train.txt", header=None, sep='\t', names=cols, index_col=0)


# In[118]:


fear_train = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/fear_train.txt", header=None, sep='\t', names=cols, index_col=0)
sadness_train = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/sadness_train.txt", header=None, sep='\t', names=cols, index_col=0)
joy_train = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/joy_.train.txt", header=None, sep='\t', names=cols, index_col=0)


# In[119]:


joy_train


# In[120]:


data = [anger_train, fear_train, sadness_train, joy_train]


# Concatinating all the datasets into one

# In[121]:


data_training = pd.concat(data)
data_training.reset_index(inplace=True)
data_training.label.value_counts()


# Countplot of emotions

# In[122]:


ax = sns.countplot(x="label", data=data_training, palette="Set3")


# Adding columns to the dataset wordcount, charcount, punctuationcount

# In[123]:


punc = string.punctuation
data_training['word_count'] = data_training['text'].apply(lambda x:len(x.split()))
data_training['char_count'] = data_training['text'].apply(lambda x:len(x.replace(' ','')))
data_training['punc_count'] = data_training['text'].apply(lambda x:len([a for a in x if a in punc]))
data_training.head()


# Box plot for anger vs intensity

# In[124]:


colors = ['red', 'black', 'lightblue', 'yellow']
bplot = sns.boxplot(data=data_training, x='label', y='intensity')
for i in range(4):
    bplot.artists[i].set_facecolor(colors[i])
plt.title('Average Intensity for Each Label')


# Relation between char_count and intensity

# In[125]:


sns.jointplot(data=data_training, x='char_count', y='intensity', kind='hex')
plt.suptitle('Relationship between char_count and intensity', y=1.08)


# In[126]:


sns.jointplot(data=data_training, x='word_count', y='intensity', kind='hex')
plt.suptitle('Relationship between word_count and intensity', y=1.08)


# In[127]:


sns.jointplot(data=data_training, x='punc_count', y='intensity', kind='hex')
plt.suptitle('Relationship between punct_count and intensity', y=1.08)


# combine all the rows of each emotion

# In[128]:


join_text_fear = ' '.join(data_training[data_training['label']=='fear']['text'].values)
join_text_anger = ' '.join(data_training[data_training['label']=='anger']['text'].values)
join_text_joy = ' '.join(data_training[data_training['label']=='joy']['text'].values)
join_text_sadness = ' '.join(data_training[data_training['label']=='sadness']['text'].values)


# In[129]:


join_text_sadness


# In[130]:


#Show 10 most used hashtag for each emotion
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
fig.subplots_adjust(hspace=.5)

fear_counter = Counter(word for word in join_text_fear.split(' ') if word != '' and word[0]=='#')
most_common = fear_counter.most_common(10)
df = pd.DataFrame(most_common, columns=["Hashtag", "Total"])
df.plot.barh(y="Total", x="Hashtag", ax=ax1, color="black",title="10 Most Used Hashtag for Fear")

anger_counter = Counter(word for word in join_text_anger.split(' ') if word != '' and word[0]=='#')
most_common = anger_counter.most_common(10)
df = pd.DataFrame(most_common, columns=["Hashtag", "Total"])
df.plot.barh(y="Total", x="Hashtag", ax=ax2, color="red",title="10 Most Used Hashtag for Anger")

joy_counter = Counter(word for word in join_text_joy.split(' ') if word != '' and word[0]=='#')
most_common = joy_counter.most_common(10)
df = pd.DataFrame(most_common, columns=["Hashtag", "Total"])
df.plot.barh(y="Total", x="Hashtag", ax=ax3, color="yellow",title="10 Most Used Hashtag for Joy")

sadness_counter = Counter(word for word in join_text_sadness.split(' ') if word != '' and word[0]=='#')
most_common = sadness_counter.most_common(10)
df = pd.DataFrame(most_common, columns=["Hashtag", "Total"])
df.plot.barh(y="Total", x="Hashtag", ax=ax4, color="blue",title="10 Most Used Hashtag for Sadness")


# In[131]:


stopwords = set(STOPWORDS)

fear_wordcloud = WordCloud(max_font_size=50, background_color='black', stopwords=stopwords, width=900, height=400).generate(join_text_fear)
anger_wordcloud = WordCloud(max_font_size=50, background_color='darkred', stopwords=stopwords, width=900, height=400).generate(join_text_anger)
joy_wordcloud = WordCloud(max_font_size=50, background_color='lightyellow', stopwords=stopwords, width=900, height=400).generate(join_text_joy)
sadness_wordcloud = WordCloud(max_font_size=50, background_color='lightblue', stopwords=stopwords, width=900, height=400).generate(join_text_sadness)


# In[132]:


fear_wordcloud


# In[134]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 25))

ax1.imshow(fear_wordcloud)
ax1.set_title("Fear", size=25)
ax1.axis('off')

ax2.imshow(anger_wordcloud)
ax2.set_title("Anger", size=25)
ax2.axis('off')

ax3.imshow(joy_wordcloud)
ax3.set_title("Joy", size=25)
ax3.axis('off')

ax4.imshow(sadness_wordcloud)
ax4.set_title("Sadness", size=25)
ax4.axis('off')


# In[137]:


clean_data_training_list = tweet_cleaner(data_training)


# tokenization and then removing all punctuations, stopwords etc

# In[138]:


from nltk.corpus import stopwords

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[A-Za-z0-9./]+'
pat3 = r'[0-9]+'
combined_pat = r'|'.join((pat1, pat2, pat3))
stop_words= set(stopwords.words('english'))

def tweet_cleaner(data_frame):
    print('Cleaning and parsing the tweets.....\n')
    clean_data = []
    for index, row in data_frame.iterrows():
        stripped = re.sub(combined_pat, '', row.text)
        lower_case = stripped.lower()
        words = tok.tokenize(lower_case)
        filtered_words = [w for w in words if not w in stop_words]
        clean_data.append((' '.join(filtered_words)).strip())
        
    print('Done!')
    return clean_data


# In[139]:


data_training.text = pd.DataFrame(clean_data_training_list)
data_training.head()


# apply bag of words and TFID technique

# In[140]:


labels = pd.get_dummies(data_training['label'])

#applying bag of words and tf-idf technique to vectorise the tweets
vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 3))
X_BoW = vectorizer.fit_transform(data_training.text)
X_BoW = pd.DataFrame.sparse.from_spmatrix(X_BoW).join(labels)

vectorizer_tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer_tfidf.fit_transform(data_training.text)
X_tfidf = pd.DataFrame.sparse.from_spmatrix(X_tfidf).join(labels)


# training the data using different models

# In[141]:


#Linear Regression
BoW_linreg = LinearRegression().fit(X_BoW, data_training['intensity'])
tfidf_linreg = LinearRegression().fit(X_tfidf, data_training['intensity'])
#Ridge Regression
BoW_ridge = Ridge().fit(X_BoW, data_training['intensity'])
tfidf_ridge = Ridge().fit(X_tfidf, data_training['intensity'])
#Knn Regression
n_neighbors=5
BoW_knn = neighbors.KNeighborsRegressor(n_neighbors,weights='uniform').fit(X_BoW, data_training['intensity'])
tfidf_knn = neighbors.KNeighborsRegressor(n_neighbors,weights='uniform').fit(X_tfidf, data_training['intensity'])
#Decision Tree Regression
BoW_tree = tree.DecisionTreeRegressor(max_depth=1).fit(X_BoW, data_training['intensity'])
tfidf_tree = tree.DecisionTreeRegressor(max_depth=1).fit(X_tfidf, data_training['intensity'])
#Support Vector Regression
BoW_svr = svm.SVR().fit(X_BoW, data_training['intensity'])
tfidf_svr = svm.SVR().fit(X_tfidf, data_training['intensity'])


# loading the dev data

# In[142]:


cols = ['id', 'text', 'label', 'intensity']


anger_dev = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/anger_dev.txt", header=None, sep='\t', names=cols, index_col=0)
fear_dev = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/fear_dev.txt", header=None, sep='\t', names=cols, index_col=0)
sad_dev = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/sadness_dev.txt", header=None, sep='\t', names=cols, index_col=0)
joy_dev = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/joy_dev.txt", header=None, sep='\t', names=cols, index_col=0)

joy_dev.head()


# In[143]:


data = [anger_dev, fear_dev, sad_dev, joy_dev]
data_dev = pd.concat(data)
data_dev.reset_index(inplace=True)
data_dev.label.value_counts()


# In[144]:


clean_data_dev_list = tweet_cleaner(data_dev)


# In[145]:


data_dev.text = pd.DataFrame(clean_data_dev_list)
data_dev.head()


# creating dummies for label feature

# In[146]:


labels = pd.get_dummies(data_dev['label'])

X_Dev_BoW = vectorizer.transform(data_dev['text'])
X_Dev_BoW = pd.DataFrame.sparse.from_spmatrix(X_Dev_BoW).join(labels)

X_Dev_tfidf = vectorizer_tfidf.transform(data_dev['text'])
X_Dev_tfidf = pd.DataFrame.sparse.from_spmatrix(X_Dev_tfidf).join(labels)


# actual intensity vs predicted intensity

# In[147]:


y_actual = data_dev['intensity']
y_predicted = tfidf_svr.predict(X_Dev_tfidf)

pd.DataFrame(data={"Actual Intensity" : data_dev['intensity'], "Predicted Intensity" : tfidf_svr.predict(X_Dev_tfidf)})


# In[148]:


sns.regplot(y_actual, y_predicted, line_kws={'color':'red'})
plt.xlabel("Actual Intensity")
plt.ylabel("Predicted Intensity")


# In[149]:


combined_training = pd.concat([data_training[['id', 'text', 'label', 'intensity']], data_dev]).reset_index()
combined_training.shape


# In[150]:


#ANGER
anger = combined_training.loc[combined_training['label']=='anger']
anger_vectorizer = TfidfVectorizer(max_features=1000)
X_anger = anger_vectorizer.fit_transform(anger['text'])
anger_model = svm.SVR().fit(X_anger, anger['intensity'])

#FEAR
fear = combined_training.loc[combined_training['label']=='fear']
fear_vectorizer = TfidfVectorizer(max_features=1000)
X_fear = fear_vectorizer.fit_transform(fear['text'])
fear_model = svm.SVR().fit(X_fear, fear['intensity'])

#SADNESS
sad = combined_training.loc[combined_training['label']=='sadness']
sad_vectorizer = TfidfVectorizer(max_features=1000)
X_sad = sad_vectorizer.fit_transform(sad['text'])
sad_model = svm.SVR().fit(X_sad, sad['intensity'])

#JOY
joy = combined_training.loc[combined_training['label']=='joy']
joy_vectorizer = TfidfVectorizer(max_features=1000)
X_joy = joy_vectorizer.fit_transform(joy['text'])
joy_model = svm.SVR().fit(X_joy, joy['intensity'])


# load test data

# In[151]:


#Load testing data
cols = ["id", "text", "label", "intensity"]
anger_test = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/anger_test.txt", header=None, sep="\t", names=cols, index_col=0)
fear_test = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/fear_test.txt", header=None, sep="\t", names=cols, index_col=0)
sad_test = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/sadness_test.txt", header=None, sep="\t", names=cols, index_col=0)
joy_test = pd.read_csv("C:/Users/nshar/OneDrive/Desktop/EMOINT/joy_test.txt", header=None, sep="\t", names=cols, index_col=0)

anger_test.head()


# In[99]:


anger_text = tweet_cleaner(anger_test)
fear_text = tweet_cleaner(fear_test)
sad_text = tweet_cleaner(sad_test)
joy_text = tweet_cleaner(joy_test)


# In[100]:


#ANGER
X_anger_test = anger_vectorizer.transform(anger_text)
Y_anger_actual = anger_test['intensity']
Y_anger_predicted = anger_model.predict(X_anger_test)

#FEAR
X_fear_test = fear_vectorizer.transform(fear_text)
Y_fear_actual = fear_test['intensity']
Y_fear_predicted = fear_model.predict(X_fear_test)

#SADNESS
X_sad_test = sad_vectorizer.transform(sad_text)
Y_sad_actual = sad_test['intensity']
Y_sad_predicted = sad_model.predict(X_sad_test)

#JOY
X_joy_test = joy_vectorizer.transform(joy_text)
Y_joy_actual = joy_test['intensity']
Y_joy_predicted = joy_model.predict(X_joy_test)


# In[101]:


#Visualising the result of the predictions
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 24))

#ANGER
sns.regplot(Y_anger_actual, Y_anger_predicted, ax=ax1, line_kws={'color':'red'}, scatter_kws={"color": "lightblue"})
ax1.set_xlabel("Actual Intensity")
ax1.set_ylabel("Predicted Intensity")
ax1.set_title("ANGER")

#FEAR
sns.regplot(Y_fear_actual, Y_fear_predicted, ax=ax2, line_kws={'color':'black'}, scatter_kws={"color": "lightblue"})
ax2.set_xlabel("Actual Intensity")
ax2.set_ylabel("Predicted Intensity")
ax2.set_title("FEAR")

#SADNESS
sns.regplot(Y_sad_actual, Y_sad_predicted, ax=ax3, line_kws={'color':'blue'}, scatter_kws={"color": "lightblue"})
ax3.set_xlabel("Actual Intensity")
ax3.set_ylabel("Predicted Intensity")
ax3.set_title("SADNESS")

#JOY
sns.regplot(Y_joy_actual, Y_joy_predicted, ax=ax4, line_kws={'color':'yellow'}, scatter_kws={"color": "lightblue"})
ax4.set_xlabel("Actual Intensity")
ax4.set_ylabel("Predicted Intensity")
ax4.set_title("JOY")


# In[43]:


#Using the official evalution function
def evaluate(pred,gold):

    # lists storing gold and prediction scores
    gold_scores=[]  
    pred_scores=[]

    # lists storing gold and prediction scores where gold score >= 0.5
    gold_scores_range_05_1=[]
    pred_scores_range_05_1=[]
        
    for p in pred:
        pred_scores.append(p)
        
    for g in gold:
        gold_scores.append(g)

    for i in range(len(gold_scores)):
        if gold_scores[i] >= 0.5:
            gold_scores_range_05_1.append(gold_scores[i])
            pred_scores_range_05_1.append(pred_scores[i])

    
    # return zero correlation if predictions are constant
    if np.std(pred_scores)==0 or np.std(gold_scores)==0:
        return (0,0,0,0)
    

    pears_corr=scipy.stats.pearsonr(pred_scores,gold_scores)[0]                                     
    pears_corr_range_05_1=scipy.stats.pearsonr(pred_scores_range_05_1,gold_scores_range_05_1)[0]                                           
    
    
    return (pears_corr,pears_corr_range_05_1)


# In[44]:


pear_results=[]
spear_results=[]

pear_results_range_05_1=[]
spear_results_range_05_1=[]

num_pairs = 4
argv = ["Anger_Actual", Y_anger_actual, "Anger_Predicted", Y_anger_predicted, "Fear_Actual", Y_fear_actual, "Fear_Predicted", Y_fear_predicted, "Sad_Actual", Y_sad_actual, "Sad_Predicted", Y_sad_predicted, "Joy_Actual", Y_joy_actual, "Joy_Predicted", Y_joy_predicted]

for i in range(0,num_pairs*4,4):
    name_gold = argv[i]
    gold=argv[i+1]
    name_pred = argv[i+2]
    pred=argv[i+3]       
    result=evaluate(pred,gold)
    
    print ("Pearson correlation between ", name_pred, " and ", name_gold, ":\t", str(result[0]))        
    pear_results.append(result[0])


    print ("Pearson correlation for gold scores in range 0.5-1 between ",name_pred," and ",name_gold,":\t",str(result[1]))       
    pear_results_range_05_1.append(result[1])


avg_pear=np.mean(pear_results)

avg_pear_range_05_1=np.mean(pear_results_range_05_1)

print ("Average Pearson correlation:\t",str(avg_pear))

print ("Average Pearson correlation for gold scores in range 0.5-1:\t", str(avg_pear_range_05_1))


# In[108]:


Y_anger_predicted


# In[ ]:




