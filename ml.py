import pandas as pd
from sqlalchemy import create_engine


# Loading cleanded table into amazon_reviews df 
engine = create_engine('database connection URL ')

table_name = 'amazon_reviews_cleaned_eng'

amazon_reviews = pd.read_sql_table(table_name, engine)

amazon_reviews.info()
amazon_reviews.shape



#### WordCloud

from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

combined_text = " ".join(review for review in amazon_reviews['combined_text'])

wordcloud = WordCloud(stopwords=set(stopwords.words('english')), background_color="white").generate(combined_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


### Stop Words 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')

custom_words = ['song', 'music', 'listen', 'one']

extended_stop_words_list = list(set(english_stop_words + custom_words))

print(extended_stop_words_list)


#### Vectorization and split 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(stop_words=extended_stop_words_list, max_features=5000, ngram_range=(1,2)) 

X = vectorizer.fit_transform(amazon_reviews['combined_text'])
y = amazon_reviews['sentiment']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



####  KNN Classifier
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, roc_curve, roc_auc_score,classification_report
knn_act = accuracy_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions, average='weighted') 
knn_rec = recall_score(y_test,knn_predictions)
knn_prec = precision_score(y_test,knn_predictions)
knn_kappa = cohen_kappa_score(y_test,knn_predictions)
knn_fpr , knn_tpr, _ = roc_curve(y_test,knn_predictions)
knn_auc = roc_auc_score(y_test, knn_predictions)

from sklearn.metrics import classification_report
print(classification_report(y_test, knn_predictions))

 

#### Logistic regresion 
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000) 
log_reg.fit(X_train, y_train)
log_reg_predictions = log_reg.predict(X_test)

print("\nLogistic Regression Classifier Evaluation")
lr_act = accuracy_score(y_test, log_reg_predictions)
lr_f1 = f1_score(y_test, log_reg_predictions, average='weighted') 
lr_rec = recall_score(y_test,log_reg_predictions)
lr_prec = precision_score(y_test,log_reg_predictions)
lr_kappa = cohen_kappa_score(y_test,log_reg_predictions)
lr_fpr , lr_tpr, _ = roc_curve(y_test,log_reg_predictions)
lr_auc = roc_auc_score(y_test, log_reg_predictions)

print(classification_report(y_test, log_reg_predictions))


######## Random Forest 
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  
random_forest.fit(X_train, y_train)
rf_predictions = random_forest.predict(X_test)

print("\nRandom Forest Classifier Evaluation")
rf_act = accuracy_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
rf_rec = recall_score(y_test,rf_predictions)
rf_prec = precision_score(y_test,rf_predictions)
rf_kappa = cohen_kappa_score(y_test,rf_predictions)
rf_fpr , rf_tpr, _ = roc_curve(y_test,rf_predictions)
rf_auc = roc_auc_score(y_test, rf_predictions)

print(classification_report(y_test, rf_predictions))


### MODEL COMPARISION 

import seaborn as sns
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')


barWidth = 0.2

rf_score = [rf_act,rf_prec,rf_rec,rf_f1,rf_kappa]
knn_score = [knn_act,knn_prec,knn_rec,knn_f1,knn_kappa]
lr_score = [lr_act,lr_prec,lr_rec,lr_f1,lr_kappa]

import numpy as np
r1 = np.arange(len(rf_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]


ax1.bar(r1, rf_score, width=barWidth, edgecolor='white', label='RF')
ax1.bar(r2, knn_score, width=barWidth, edgecolor='white', label='KNN')
ax1.bar(r3, lr_score, width=barWidth, edgecolor='white', label='LR')

ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(rf_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)


ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()


ax2.plot(rf_fpr, rf_tpr, label='Random Forest, auc = {:0.5f}'.format(rf_auc))
ax2.plot(knn_fpr, knn_tpr, label='KNN, auc = {:0.5f}'.format(knn_auc))
ax2.plot(lr_fpr,lr_tpr, label='Logistic Regression, auc = {:0.5f}'.format(lr_auc))


ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')

ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()
