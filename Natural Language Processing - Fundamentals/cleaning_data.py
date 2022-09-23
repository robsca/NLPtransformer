'''CLEANING DATA'''

import pickle
import pandas as pd

# Load transcriptions data
comedians = ['einstein', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']
data = {}
for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)


#This function will take the values in the dictionary(list of words) and transform them to strings
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

# Transform in DataFrame
def transform_in_dataframe(data_combined):
    data_df = pd.DataFrame.from_dict(data_combined).transpose()
    data_df.columns = ['transcript']
    data_df = data_df.sort_index()
    return data_df

data_combined = {key: [combine_text(value)] for (key, value) in data.items()} # transform script in string type
data_df = transform_in_dataframe(data_combined)  # transform dictionary in DataFrame

# add names in column
full_names = ['einstein', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj','Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
data_df['full_name'] = full_names 
# save file
data_df.to_pickle("corpus.pkl") 

'''
2. Cleaning The Data
When dealing with numerical data, data cleaning often involves removing null values and duplicate data, dealing with outliers, etc.
With text data, there are some common data cleaning techniques, which are also known as text pre-processing techniques.
With text data, this cleaning process can go on forever. There's always an exception to every cleaning step.
So, we're going to follow the MVP (minimum viable product) approach - start simple and iterate. 
Here are a bunch of things you can do to clean your data. 
We're going to execute just the common cleaning steps here and the rest can be done at a later point to improve our results.

1. Make text all lower case
2. Remove punctuation
3. Remove numerical values
4. Remove common non-sensical text
5. Tokenize text
6. Remove stop words
'''

# Apply a first round of text cleaning techniques
import re
import string

''' Cleaning Functions'''
def remove_text_squarebrackets_punctuation_numbers(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\ [ . * ? , \ ] ( ) \n "" "  ' , '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('1234567890','',text)
    return text
round1 = lambda x: remove_text_squarebrackets_punctuation_numbers(x) # define function
data_clean = pd.DataFrame(data_df.transcript.apply(round1)) # apply changes

# Apply a second round of cleaning
def remove_punctuation_left_and_nonsense_text(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time.'''
    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = re.sub('\n', '', text)
    text = text.encode("utf8").decode("ascii", 'ignore')
    return text
    
round2 = lambda x: remove_punctuation_left_and_nonsense_text(x) # define function
data_clean = pd.DataFrame(data_clean.transcript.apply(round2)) # apply changes

'''
Document-Term Matrix
For many of the techniques we'll be using in future notebooks, the text must be tokenized, meaning broken down into smaller pieces. The most common tokenization technique is to break down text into words. We can do this using scikit-learn's CountVectorizer, where every row will represent a different document and every column will represent a different word.
In addition, with CountVectorizer, we can remove stop words. Stop words are common words that add no additional meaning to text such as 'a', 'the', etc.
'''
# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

# Let's pickle it for later use
data_dtm.to_pickle("document-term-matrix.pkl")

# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("counter_vectorizer.pkl", "wb"))