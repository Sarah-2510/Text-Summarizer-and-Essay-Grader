from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd
stop_words=set(stopwords.words('english'))
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle
count_fit=pd.read_csv(r'count_fit_data.csv')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd
stop_words=set(stopwords.words('english'))
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle
count_fit=pd.read_csv(r'count_fit_data.csv')

def grade(essay):
    
    
    essay=re.sub("[^A-Za-z ]","",essay)
    x=[]
    for i in essay.split():
        
        if i.startswith("@") and i in stop_words:
            continue
        else:
            x.append(i)        
    essay=' '.join(x).lower()
    essay_df=pd.DataFrame({'essay':[essay]})
    #return essay_df


    def sent2word(x):
        x=re.sub("[^A-Za-z0-9]"," ",x)
        words=nltk.word_tokenize(x)
        return words

    def essay2word(essay):
        essay = essay.strip()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw = tokenizer.tokenize(essay)
        final_words=[]
        for i in raw:
            if(len(i)>0):
                final_words.append(sent2word(i))
        return final_words

    def noOfWords(essay):
        count=0
        for i in essay2word(essay):
            count=count+len(i)
        return count

    def noOfChar(essay):
        count=0
        for i in essay2word(essay):
            for j in i:
                count=count+len(j)
        return count

    def avg_word_len(essay):
        return noOfChar(essay)/noOfWords(essay)

    def noOfSent(essay):
        return len(essay2word(essay))

    def count_pos(essay):
        sentences = essay2word(essay)
        noun_count=0
        adj_count=0
        verb_count=0
        adverb_count=0
        for i in sentences:
            pos_sentence = nltk.pos_tag(i)
            for j in pos_sentence:
                pos_tag = j[1]
                if(pos_tag[0]=='N'):
                    noun_count+=1
                elif(pos_tag[0]=='V'):
                    verb_count+=1
                elif(pos_tag[0]=='J'):
                    adj_count+=1
                elif(pos_tag[0]=='R'):
                    adverb_count+=1
        return noun_count,verb_count,adj_count,adverb_count

    pro_test = essay_df.copy()
    pro_test['char_count'] = pro_test['essay'].apply(noOfChar)
    pro_test['word_count'] = pro_test['essay'].apply(noOfWords)
    pro_test['sent_count'] = pro_test['essay'].apply(noOfSent)
    pro_test['avg_word_len'] = pro_test['essay'].apply(avg_word_len)

    pro_test['noun_count'], pro_test['adj_count'], pro_test['verb_count'], pro_test['adv_count'] = zip(*pro_test['essay'].map(count_pos))
    #pro_data.to_csv("Processed_data.csv")

    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    count_vectorfit = vectorizer.fit(count_fit['clean_essay'])
    count_transform=vectorizer.transform(pro_test['essay'])
    feature_names = vectorizer.get_feature_names()
    x = count_transform.toarray()
    X_test = np.concatenate((pro_test.iloc[:, 2:], pd.DataFrame(x)), axis = 1)

    model=pickle.load(open("svr_pp",'rb'))
    ypred=model.predict(X_test)
    return ypred
