# -*- coding: utf-8 -*-
# NLP 튜토리얼 - 노래 가사 분석 및 검색 엔진 만들기

# 1. 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 불용어 리스트 정의
def load_stopwords():
    # 기본 불용어 리스트
    stopwords = [
        '은', '는', '이', '가', '을', '를', '에', '에서', '의', '로', '으로', '와', '과', '도', '만', '까지',
        '에서', '에게', '하고', '이다', '있다', '없다', '같다', '있다', '그', '저', '이', '것', '수', '등', '들',
        '때', '한', '지', '하', '오', '말', '일', '때문', '거', '게', '너무', '더', '나', '내', '걸', '이런',
        '저런', '왜', '그냥', '다시', '정도', '때문', '이제', '다시', '모두', '아니', '없이', '같이', '처럼',
        '다른', '모든', '우리', '내가', '네가', '그녀', '그들', '나의', '너의', '이것', '저것', '그것', '누구',
        '무엇', '어디', '언제', '어떻게', '왜', '몇', '얼마나', '모든', '다른', '어떤', '무슨', '아무', '이런',
        '저런', '그런', '무슨', '아무', '이런', '저런', '그런', '무슨', '아무', '이런', '저런', '그런', '무슨', '아무',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
        "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
        'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
        'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
        'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    ]
    return set(stopwords)

# 2. 데이터 로드 및 탐색
print("1. 데이터 로드 중...")
df = pd.read_csv('songs.csv')
print("\n데이터 샘플:")
print(df.head())
print("\n데이터 정보:")
df.info()

# 3. 텍스트 데이터 전처리
def preprocess_text(text, remove_stopwords=True):
    # 한글, 영어, 공백만 남기고 제거
    import re
    text = re.sub(r'[^가-힣a-zA-Z\s]', ' ', str(text))
    
    # 불용어 제거
    if remove_stopwords:
        stopwords = load_stopwords()
        words = text.split()
        words = [word for word in words if word not in stopwords]
        text = ' '.join(words)
    
    return text.strip()

print("\n2. 텍스트 전처리 중...")
# 불용어 제거를 포함한 텍스트 전처리
df['cleaned_lyrics'] = df['lyrics'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

# 4. 형태소 분석기 초기화
tokenizer = Okt()

def get_nouns(text, min_length=1):
    # 명사만 추출하고 길이 제한 적용
    nouns = tokenizer.nouns(text)
    # 최소 길이 이상의 명사만 필터링
    nouns = [noun for noun in nouns if len(noun) >= min_length]
    return ' '.join(nouns)

print("\n3. 명사 추출 중...")
# 최소 2글자 이상의 명사만 추출
df['nouns'] = df['cleaned_lyrics'].apply(lambda x: get_nouns(x, min_length=2))

# 5. BOW (Bag of Words) 구현
def create_bow(texts):
    # 단어 사전 생성
    words = ' '.join(texts).split()
    word_counts = Counter(words)
    vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
    
    # BOW 벡터 생성
    bow_vectors = []
    for text in texts:
        vector = [0] * len(vocab)
        for word in text.split():
            if word in vocab:
                vector[vocab[word]] += 1
        bow_vectors.append(vector)
    
    return bow_vectors, vocab

print("\n4. BOW 생성 중...")
bow_vectors, vocab = create_bow(df['nouns'])
print(f"\n어휘 크기: {len(vocab)}")
print("\nBOW 벡터 샘플 (첫 번째 문서):")
print(bow_vectors[0][:20])  # 첫 번째 문서의 BOW 벡터 일부 출력

# 6. TF-IDF 구현
print("\n5. TF-IDF 계산 중...")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['nouns'])
print("\nTF-IDF 행렬 크기:", tfidf_matrix.shape)

# 7. 단어 구름 시각화
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc'  # macOS 한글 폰트 경로
                        ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

print("\n6. 단어 구름 생성 중...")
all_nouns = ' '.join(df['nouns'])
generate_wordcloud(all_nouns)

# 8. 간단한 검색 엔진 구현
class SimpleSearchEngine:
    def __init__(self, df, tfidf_matrix, vectorizer):
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
    
    def search(self, query, top_n=5):
        # 쿼리 전처리 및 변환
        query = preprocess_text(query)
        query_nouns = get_nouns(query)
        query_vec = self.vectorizer.transform([query_nouns])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 상위 N개 결과 반환
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'title': self.df.iloc[idx]['title'],
                'artist': self.df.iloc[idx]['artist'],
                'similarity': f"{similarities[idx]:.2%}",
                'lyrics_preview': self.df.iloc[idx]['lyrics'][:100] + '...'
            })
        
        return pd.DataFrame(results)

# 9. 검색 엔진 초기화 및 테스트
print("\n7. 검색 엔진 초기화 중...")
search_engine = SimpleSearchEngine(df, tfidf_matrix, tfidf_vectorizer)

# 테스트 검색
print("\n테스트 검색: '사랑'")
results = search_engine.search('사랑')
print("\n검색 결과:")
print(results[['title', 'artist', 'similarity']])
