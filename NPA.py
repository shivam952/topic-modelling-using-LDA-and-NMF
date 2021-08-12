# We use Non-Negative Matrix Factorization
import pandas as pd
quora = pd.read_csv('quora_questions.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = tfidf.fit_transform(quora['Question'])

from sklearn.decomposition import NMF
nmf_model = NMF(n_components=20, random_state =42)
nmf_model.fit(dtm)

for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
topic_results = nmf_model.transform(dtm)    

quora['Topic'] = topic_results.argmax(axis=1)

