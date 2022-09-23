'''
Gather data with webscraper
'''
import requests 
from bs4 import BeautifulSoup
import pickle

# scrapes transcripts
def url_to_transcript(url):

    page = requests.get(url).text
    soup = BeautifulSoup(page,"lxml")
    text = []
    for p in soup.find_all('p'):
        text.append(p.text) 
    print(url)
    return text

# URLs of transcripts in scope
urls = [
    'https://en.wikiquote.org/wiki/Albert_Einstein',
    'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
    'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
    'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
    'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
    'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
    'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
    'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
    'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
    'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
    'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
    'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/'
        ]

# Comedian names
comedians = ['einstein', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']


transcripts = [url_to_transcript(u) for u in urls]
print(transcripts)

#Pickle files for later use
# Make a new directory to hold the text files

for i, c in enumerate(comedians):
     with open("transcripts/" + c + ".txt", "wb") as file:
        pickle.dump(transcripts[i], file)
