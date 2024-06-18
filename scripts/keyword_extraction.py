from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(texts, top_n=10):
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    terms = tfidf.get_feature_names_out()

    keywords = []
    for row in tfidf_matrix:
        indices = row.toarray().argsort()[-top_n:]
        keywords.append([terms[i] for i in indices])

    return keywords
