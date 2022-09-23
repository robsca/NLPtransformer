# Read in the corpus, including punctuation!
import pandas as pd

data = pd.read_pickle('corpus.pkl')
data

# Extract only Ali Wong's text
einstein_text = data.transcript.loc['einstein']

## Build a Markov Chain Function
from collections import defaultdict

def markov_chain(text):
    '''The input is a string of text and the output will be a dictionary with each word as
    a key and each value as the list of words that come after the key in the text.'''
    
    # Tokenize the text by word, though including punctuation
    words = text.split(' ')
    
    # Initialize a default dictionary to hold all of the words and next words
    m_dict = defaultdict(list)
    
    # Create a zipped list of all of the word pairs and put them in word: list of next words format
    for current_word, next_word in zip(words[0:-1], words[1:]):
        m_dict[current_word].append(next_word)

    # Convert the default dict back into a dictionary
    m_dict = dict(m_dict)
    return m_dict

# Create the dictionary for Einstein's routine, take a look at it
einstein_dict = markov_chain(einstein_text)
print(f'lunghezza dizionario {len(einstein_dict)}')
for key, value in einstein_dict.items():
    print(key, value)

import random
def generate_sentence(chain, count=60):
    '''Input a dictionary in the format of key = current word, value = list of next words
        along with the number of words you would like to see in your generated sentence.'''
 
    # Capitalize the first word
    word1 = random.choice(list(chain.keys()))
    sentence = word1.capitalize()

    # Generate the second word from the value list. Set the new word as the first word. Repeat.
    for i in range(count-1):
        word2 = random.choice(chain[word1])
        word1 = word2
        sentence += ' ' + word2

    # End it with a period
    sentence += '.'
    return(sentence)

sentence = (generate_sentence(einstein_dict))
print(sentence)



#################################### Speaking
import pyttsx3

engine = pyttsx3.init()
""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
engine.setProperty('rate', 150)     # setting up new voice rate
"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
engine.setProperty('volume',0.8)    # setting up volume level  between 0 and 1
"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male 1 female

engine.say(sentence)
engine.runAndWait()
