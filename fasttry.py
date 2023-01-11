from fastapi import FastAPI, File, UploadFile
import shutil
import pandas as pd
from typing import List
from fastapi.responses import HTMLResponse
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import pickle



app = FastAPI()


@app.post("/files/")
async def create_files(files: List[bytes] = File()):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile]):
    dfs=[]
    for file in files:
        filepath = file.filename

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
                dfs.append(df)
            else:
                df = pd.read_excel(filepath)
        except:
            return 401, "File is not proper"

    true=dfs[0]
    fake=dfs[1]

    true['label'] = 1
    fake['label'] = 0

    frames = [true.loc[:5000][:], fake.loc[:5000][:]]
    df = pd.concat(frames)

    X = df.drop('label', axis=1) 
    y = df['label']
    # Delete missing data
    df = df.dropna()
    df2 = df.copy()
    df2.reset_index(inplace=True)

    # nltk.download('stopwords')
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(df2)):
        review = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
    X = tfidf_v.fit_transform(corpus).toarray()
    y = df2['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pickle.dump(tfidf_v, open('tfidfvect2.pkl', 'wb'))
    train=[]
    train.append(X_train)
    train.append(X_test)
    train.append(y_train)
    train.append(y_test)

    pickle.dump(train,open('trainsets.pkl','wb'))

    return len(train)

    