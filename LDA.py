# We use Latent dirilicht Allocation
import pandas as pd
npr = pd.read_csv('npr.csv')

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = cv.fit_transform(npr['Article'])

from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)

print(len(cv.get_feature_names()))

len(LDA.components_)

single_topic = LDA.components_[0]

top_word_indices = single_topic.argsort()[-10:]

for index in top_word_indices:
    print(cv.get_feature_names()[index])

for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
topic_results= LDA.transform(dtm)

topic_results[0].argmax()

npr['Topic'] = topic_results.argmax(axis=1)
    
