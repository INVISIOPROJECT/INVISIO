
import requests
from datetime import date, timedelta

API_KEY = "ad36b4e0ebac4c86a8302bea77af8255"
topic = "artificial intelligence"
language = "en"
from_date = (date.today() - timedelta(days=7)).isoformat()  # آخر 7 أيام
sources = "bbc-news,cnn,reuters"

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={topic}&"
    f"language={language}&"
    f"from={from_date}&"
    f"sources={sources}&"
    f"sortBy=publishedAt&"
    f"pageSize=20&"
    f"apiKey={API_KEY}"
)

response = requests.get(url)
data = response.json()

if "articles" in data:
    for idx, article in enumerate(data["articles"], 1):
        print(f"🔹 {idx}. {article['title']}")
        print(f"   المصدر: {article['source']['name']}")
        print(f"   التاريخ: {article['publishedAt']}")
        print(f"   الوصف: {article.get('description', 'لا يوجد')}")
        print()
else:
    print("حدث خطأ:", data.get("message", "مشكلة غير معروفة"))

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # إزالة الرموز
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

processed_texts = [preprocess(text) for text in texts]
print(processed_texts[0])

from gensim import corpora

# إنشاء قاموس من الكلمات الفريدة
dictionary = corpora.Dictionary(processed_texts)

# تحويل النصوص إلى تمثيل رقمي (Bag-of-Words)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

!pip install gensim

from gensim.models import CoherenceModel

coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
score = coherence_model.get_coherence()
print(f"Coherence Score: {score}")

import pyLDAvis
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()  # لجوبتر نوتبوك
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)  # تفتح نافذة تفاعلية بالمتصفح



