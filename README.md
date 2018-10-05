# Fake_or_Truth_NLP
This project was implemented to try indentify fake or true posts/news about president running of USA.



```python
#Importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
```


```python
df =  pd.read_csv('C:\\Users\\filipe.luz\\Desktop\\TCC -MBA\\fake_or_real_news.csv', sep=',')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6335 entries, 0 to 6334
    Data columns (total 4 columns):
    Unnamed: 0    6335 non-null int64
    title         6335 non-null object
    text          6335 non-null object
    label         6335 non-null object
    dtypes: int64(1), object(3)
    memory usage: 198.0+ KB
    


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6330</th>
      <td>4490</td>
      <td>State Department says it can't find emails fro...</td>
      <td>The State Department told the Republican Natio...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>6331</th>
      <td>8062</td>
      <td>The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...</td>
      <td>The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6332</th>
      <td>8622</td>
      <td>Anti-Trump Protesters Are Tools of the Oligarc...</td>
      <td>Anti-Trump Protesters Are Tools of the Oligar...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6333</th>
      <td>4021</td>
      <td>In Ethiopia, Obama seeks progress on peace, se...</td>
      <td>ADDIS ABABA, Ethiopia —President Obama convene...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>6334</th>
      <td>4330</td>
      <td>Jeb Bush Is Suddenly Attacking Trump. Here's W...</td>
      <td>Jeb Bush Is Suddenly Attacking Trump. Here's W...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Evaluating the model

# Create a series to store the labels: y
y = df['label']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'],y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train.values)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test.values)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

```

    ['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']
    


```python
###TfidfVectorizer for text classification###

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test.values)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train[:5].A)

```

    ['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    


```python
###Inspecting the vectors###
'''Let's inspect the vectors as a DataFrame'''

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns= count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns (Test if the column names are the same for each DataFrame): difference
difference = set(tfidf_df.columns) - set(count_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

```

       00  000  0000  00000031  000035  00006  0001  0001pt  000ft  000km  ...    \
    0   0    0     0         0       0      0     0       0      0      0  ...     
    1   0    0     0         0       0      0     0       0      0      0  ...     
    2   0    0     0         0       0      0     0       0      0      0  ...     
    3   0    0     0         0       0      0     0       0      0      0  ...     
    4   0    0     0         0       0      0     0       0      0      0  ...     
    
       حلب  عربي  عن  لم  ما  محاولات  من  هذا  والمرضى  ยงade  
    0    0     0   0   0   0        0   0    0        0      0  
    1    0     0   0   0   0        0   0    0        0      0  
    2    0     0   0   0   0        0   0    0        0      0  
    3    0     0   0   0   0        0   0    0        0      0  
    4    0     0   0   0   0        0   0    0        0      0  
    
    [5 rows x 56922 columns]
        00  000  0000  00000031  000035  00006  0001  0001pt  000ft  000km  ...    \
    0  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    1  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    2  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    3  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    4  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...     
    
       حلب  عربي   عن   لم   ما  محاولات   من  هذا  والمرضى  ยงade  
    0  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    1  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    2  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    3  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    4  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  
    
    [5 rows x 56922 columns]
    set()
    False
    


```python
#Training and testing the "fake news" model with CountVectorizer


# Import the necessary modules
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

```

    0.893352462936394
    [[ 865  143]
     [  80 1003]]
    


```python
#Training and testing the "fake news" model with TfidfVectorizer

# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE','REAL'])
print(cm)

```

    0.8565279770444764
    [[ 739  269]
     [  31 1052]]
    


```python
'''The next steps will be improve the model, to do it we have a lot of possibilities as well as
   - Tweaking alpha levels
   - Trying a new classification model
   - Training on a large dataset
   - Improving text preprocessing
   
'''

# Create the list of alphas: alphas
alphas = np.arange(0, 1, .1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
    

```

    Alpha:  0.0
    Score:  0.8813964610234337
    
    Alpha:  0.1
    

    D:\Continuum\anaconda3\lib\site-packages\sklearn\naive_bayes.py:472: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
      'setting alpha = %.1e' % _ALPHA_MIN)
    

    Score:  0.8976566236250598
    
    Alpha:  0.2
    Score:  0.8938307030129125
    
    Alpha:  0.30000000000000004
    Score:  0.8900047824007652
    
    Alpha:  0.4
    Score:  0.8857006217120995
    
    Alpha:  0.5
    Score:  0.8842659014825442
    
    Alpha:  0.6000000000000001
    Score:  0.874701099952176
    
    Alpha:  0.7000000000000001
    Score:  0.8703969392635102
    
    Alpha:  0.8
    Score:  0.8660927785748446
    
    Alpha:  0.9
    Score:  0.8589191774270684
    
    


```python
#Inspecting your model

# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0],feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])


```

    FAKE [(-11.316312804238807, '0000'), (-11.316312804238807, '000035'), (-11.316312804238807, '0001'), (-11.316312804238807, '0001pt'), (-11.316312804238807, '000km'), (-11.316312804238807, '0011'), (-11.316312804238807, '006s'), (-11.316312804238807, '007'), (-11.316312804238807, '007s'), (-11.316312804238807, '008s'), (-11.316312804238807, '0099'), (-11.316312804238807, '00am'), (-11.316312804238807, '00p'), (-11.316312804238807, '00pm'), (-11.316312804238807, '014'), (-11.316312804238807, '015'), (-11.316312804238807, '018'), (-11.316312804238807, '01am'), (-11.316312804238807, '020'), (-11.316312804238807, '023')]
    REAL [(-7.742481952533027, 'states'), (-7.717550034444668, 'rubio'), (-7.703583809227384, 'voters'), (-7.654774992495461, 'house'), (-7.649398936153309, 'republicans'), (-7.6246184189367, 'bush'), (-7.616556675728882, 'percent'), (-7.545789237823644, 'people'), (-7.516447881078008, 'new'), (-7.448027933291952, 'party'), (-7.4111484102034755, 'cruz'), (-7.410910239085596, 'state'), (-7.35748985914622, 'republican'), (-7.33649923948987, 'campaign'), (-7.2854057032685775, 'president'), (-7.2166878130917755, 'sanders'), (-7.108263114902301, 'obama'), (-6.72477133248804, 'clinton'), (-6.565395438992684, 'said'), (-6.328486029596207, 'trump')]
    
