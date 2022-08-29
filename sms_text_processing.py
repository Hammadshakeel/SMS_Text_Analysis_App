# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 14:24:29 2022

@author: Hammad khan
"""

import streamlit as st
import pandas as pd
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def main():
    st.title('SMS_Data_Processing_Text')
    df = pd.read_csv('SMS_data.csv', encoding= 'unicode_escape')
    df['Date_Received'] = pd.to_datetime(df['Date_Received'])
    PUNCT_TO_REMOVE = string.punctuation
    def remove_punctuation(Message_body):
        
        """custom function to remove the punctuation"""
        return Message_body.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    df["Message_body_punctuat"] = df["Message_body"].apply(lambda Message_body: remove_punctuation(Message_body).lower())
    #df.head()
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(Message_body):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(Message_body).split() if word not in STOPWORDS])

    df["Message_body_stop"] = df["Message_body_punctuat"].apply(lambda Message_body: remove_stopwords(Message_body))
    #df.head()
    
    cnt = Counter()
    for text in df["Message_body_stop"].values:
        for word in text.split():
            cnt[word] += 1
    cnt.most_common(10)
    
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    def remove_freqwords(text):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])

    df["Message_body_stopfreq"] = df["Message_body_stop"].apply(lambda Message_body: remove_freqwords(Message_body))
    #df.head()
    n_rare_words = 10
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    def remove_rarewords(Message_body):
        """custom function to remove the rare words"""
        return " ".join([word for word in str(Message_body).split() if word not in RAREWORDS])

    df["Message_body_stopfreqrare"] = df["Message_body_stopfreq"].apply(lambda Message_body: remove_rarewords(Message_body).lower())
    #df.head()
   
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    def lemmatize_words(Message_body_stopfreqrare):
        pos_tagged_Message_body_stopfreqrare = nltk.pos_tag(Message_body_stopfreqrare.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_Message_body_stopfreqrare])
    df["Message_body_lemmatized"] = df["Message_body_stopfreqrare"].apply(lambda Message_body_stopfreqrare: lemmatize_words(Message_body_stopfreqrare))
    #df.head()
    
    category = st.selectbox('Type of Message', df['Label'].unique())
    #date = st.selectbox('Type of Message', df['Label'].unique())
    
    cnt = Counter()
    for txt in df[df['Label'] == category]['Message_body_lemmatized'].values:
        for word in txt.split():
            cnt[word] += 1        
    cnt.most_common(10) 
    
    dfn = pd.DataFrame(cnt.most_common(10))
    
    dtr = df[df['Label']==category]
    dtr = dtr.groupby(dtr['Date_Received'].dt.month_name())['Label'].count()
    dtr = pd.DataFrame(dtr)
  
    def countplt():
        figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        # plot1 definition
        sns.barplot(x=dfn[1], y=dfn[0], ax=ax1)
        ax1.set(title='Most Common 10 words after all cleaning)', xlabel='Word-Count', ylabel='Word')
        # plt.legend()
        #st.pyplot(plt1)
        
        # line plot2 definition
        sns.lineplot(data=dtr, x=dtr.index , y=dtr['Label'])
        ax2.set_title(f'Number of {category} Messages Across Months')
        #plt.xticks(rotation=45,horizontalalignment='right',fontweight='light',fontsize='medium')
        plt.xticks(rotation=45,fontweight='light',fontsize='small')
        ax2.set_xlabel('Months')
        ax2.set_ylabel('Counts')
        
        st.pyplot(figure)
    
    
    # button to show results
    button = st.button('Show Results')
    if button:
        countplt()
        
if __name__=='__main__':
    main()
