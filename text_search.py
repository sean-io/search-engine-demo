# create class utilizing search logic

class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
