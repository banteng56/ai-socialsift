import os
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from model_hmm12 import HMM
from hmm_statement import HMM2
from fastapi.middleware.cors import CORSMiddleware
import collections
from nltk.tokenize import word_tokenize
import nltk
from datasets import load_dataset
import dill
import numpy as np
import requests
from fuzzywuzzy import process
from time import time
import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
nltk.download('punkt')

UPLOAD_FOLDER = "output/upload"
PROCESSED_FOLDER = "output/processed"
GRAPH_FOLDER="..//public"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)
if not os.path.exists(GRAPH_FOLDER):
    os.makedirs(GRAPH_FOLDER)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset = load_dataset("grit-id/id_nergrit_corpus", "ner", trust_remote_code=True)
label = dataset['train'].features['ner_tags'].feature.names

train_sentences = [example["tokens"] for example in dataset["train"]]
train_tags = [example["ner_tags"] for example in dataset["train"]]

hmm_ner_tfidf = HMM(label)
hmm_ner_tfidf.train(train_sentences, train_tags)

dataset2 = load_dataset("grit-id/id_nergrit_corpus", "statement", trust_remote_code=True)
label2 = dataset2['train'].features['ner_tags'].feature.names
train_sentences2= [example["tokens"] for example in dataset2["train"]]
train_tags2= [example["ner_tags"] for example in dataset2["train"]]
hmm_ner_tfidf2 = HMM2(label2)
hmm_ner_tfidf2.train(train_sentences2, train_tags2)

with open('naive_bayes_model.dill', 'rb') as f:
    model_data = dill.load(f)
    vectorizer = model_data['vectorizer']
    class_priors = model_data['model']['class_priors']
    word_given_class = model_data['model']['word_given_class']
    total_words_per_class = model_data['model']['total_words_per_class']
    smoothed_word_given_class = model_data['model']['smoothed_word_given_class']
    
with open('naive_bayes_model_posneg.dill', 'rb') as f:
    sentiment_model_data = dill.load(f)
    sentiment_vectorizer = sentiment_model_data['vectorizer']
    sentiment_class_priors = sentiment_model_data['model']['class_priors']
    sentiment_word_given_class = sentiment_model_data['model']['word_given_class']
    sentiment_smoothed_word_given_class = sentiment_model_data['model']['smoothed_word_given_class']


def cleansing(text, stopword_list=None):
    word_list = word_tokenize(text)
    word_list = [word for word in word_list if len(word) > 2 and word.isalnum()]
    if stopword_list:
        word_list = [word for word in word_list if word not in stopword_list]
    return ' '.join(word_list)

def get_frequent_words(df, column_name='Content', threshold=800):
    df[column_name] = df[column_name].astype(str).fillna("")
    text = " ".join(df[column_name])
    word_list = word_tokenize(text)
    word_count = dict(collections.Counter(word_list))
    word_freq_df = pd.DataFrame(data={'word': list(word_count.keys()), 'freq': list(word_count.values())})
    stopwords = list(word_freq_df[word_freq_df['freq'] > threshold]['word'])
    return stopwords

def combine_entities(labeled_entities):
    entities = []
    current_entity = []

    for token, tag in labeled_entities:
        if tag == "B-PER":  
            if current_entity:  
                entities.append(" ".join(current_entity))
            current_entity = [token]  
        elif tag == "I-PER" and current_entity: 
            current_entity.append(token)
    
    if current_entity: 
        entities.append(" ".join(current_entity))

    return entities

def choose_best_entity(predicted_entities):
    if not predicted_entities:  
        return None 
    
    word_counts = collections.Counter(predicted_entities)

    sorted_entities = sorted(word_counts.items(), key=lambda x: (-len(x[0]), -x[1]))

    return sorted_entities[0][0]  

def label_and_combine_entities(tokens, model):
    predicted_tags = model.viterbi(tokens)
    
    all_tokens_with_tags = [(token, tag) for token, tag in zip(tokens, predicted_tags)]
    
    graph_res = [(token, tag) for token, tag in all_tokens_with_tags if tag in {"B-PER","I-PER"}]
    
    person_entities = [(token, tag) for token, tag in all_tokens_with_tags if tag in {"B-PER", "I-PER"}]
    
    combined_person_entities = combine_entities(person_entities)
    
    best_entity = choose_best_entity(combined_person_entities)
    
    return best_entity, graph_res


def preprocess_and_label(df, model, column_name='Content', stopword_threshold=800):
    df = df.dropna(subset=[column_name]) 
    df = df.drop_duplicates(subset=[column_name])  
    
    stopwords = get_frequent_words(df, column_name, threshold=stopword_threshold)
    
    df['Content_Cleaned'] = df[column_name].apply(lambda x: cleansing(x, stopwords))
    df['Content_Tokens'] = df['Content_Cleaned'].apply(lambda x: word_tokenize(x))
    
    df[['Predicted_Entities', 'Graph_Res']] = df['Content_Tokens'].apply(lambda tokens: label_and_combine_entities(tokens, model)).apply(pd.Series)
    
    return df[['Content', 'Content_Cleaned', 'Content_Tokens', 'Predicted_Entities', 'Graph_Res']]
def get_hmm2_statement(tokens):
    try:
        predicted_tags = hmm_ner_tfidf2.viterbi(tokens)
        
        statement_entities = []
        current_entity = [] 
        for token, tag in zip(tokens, predicted_tags):
            if tag == "B-STAT" or tag == "I-STAT":
                if tag == "B-STAT" and current_entity:
                    
                    statement_entities.append((" ".join(current_entity), "B-STAT"))
                    current_entity = []  
                current_entity.append(token)  
            else:
                if current_entity:
                    statement_entities.append((" ".join(current_entity), "B-STAT"))
                    current_entity = []  

        if current_entity:
            statement_entities.append((" ".join(current_entity), "B-STAT"))
        
        return statement_entities

    except Exception as e:
        print(f"Error in get_hmm2_statement: {e}")
        return None



def naive_bayes_predict(X_test):
    predictions = []
    for test_point in X_test:
        class_probabilities = {}
        for c in class_priors:
            log_prob = np.log(class_priors[c])
            for i in range(len(test_point)):
                if test_point[i] > 0:
                    log_prob += test_point[i] * np.log(smoothed_word_given_class[c][i])
            class_probabilities[c] = log_prob

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        predictions.append(predicted_class)

    return predictions
def naive_bayes_sentiment_predict(X_test):
    predictions = []
    for test_point in X_test:
        class_probabilities = {}
        for c in sentiment_class_priors:
            log_prob = np.log(sentiment_class_priors[c])
            for i in range(len(test_point)):
                if test_point[i] > 0:
                    log_prob += test_point[i] * np.log(sentiment_smoothed_word_given_class[c][i])
            class_probabilities[c] = log_prob

        predicted_class = max(class_probabilities, key=class_probabilities.get)
        predictions.append(predicted_class)

    return predictions

PROXY = {
    'http': 'http://fcPf8BsBssS89BLa:ne28lkJxVmAdDVmr_session-90VG46uO_lifetime-1s@geo.iproyal.com:12321'  
}

session = requests.Session()
session.proxies = PROXY 

def get_info_from_wikipedia(person_name):
    url = f"https://id.wikipedia.org/w/api.php"
    
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': person_name,
        'srprop': 'snippet',  
    }
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        search_results = data['query']['search']
        
        if search_results:
            article_titles = [result['title'] for result in search_results]
            best_match = process.extractOne(person_name, article_titles)
            return best_match
        else:
            return None
    except requests.exceptions.RequestException as e:
        return None 

def process_csv_and_add_descriptions(df):
    valid_results = []
    start_time = time() 

    for index, row in df.iterrows():
        person_name = row['Predicted_Entities']
        
        if pd.isna(person_name):
            person_name = ""  
        person_name = str(person_name) 

        if person_name:
            best_match = get_info_from_wikipedia(person_name)
            if best_match:
                if best_match[1] >= 60:
                    if best_match[0] not in valid_results:
                        valid_results.append(best_match[0])
                        df.at[index, 'Predicted_Entities'] = best_match[0]
            else:
                df.at[index, 'Predicted_Entities'] = row['Predicted_Entities']
        else:
            df.at[index, 'Predicted_Entities'] = None

    for index, row in df.iterrows():
        person_name = row['Predicted_Entities']
        
        if isinstance(person_name, str) and person_name.strip() != "":
            for valid_name in valid_results:
                match_score = process.extractOne(person_name, [valid_name])
                if match_score and match_score[1] >= 60: 
                    df.at[index, 'Predicted_Entities'] = valid_name
                    break  

    end_time = time() 
    print(f"Proses selesai. Waktu proses: {end_time - start_time:.2f} detik")
    return df
def process_csv(file_path):
    df = pd.read_csv(file_path)
    texts = df['Content_Cleaned']

    global_tag_dict = {}

    for graph_res in df['Combined_Res']:
        ner_result = eval(graph_res)
        current_entity = None
        current_tag = None

        for word, tag in ner_result:
            if tag.startswith("B-"):
                if current_entity:
                    if current_tag not in global_tag_dict:
                        global_tag_dict[current_tag] = []
                    global_tag_dict[current_tag].append(" ".join(current_entity))
                if tag != 'B-GPE':  
                    current_entity = [word]
                    current_tag = tag
                else:
                    current_entity = None
            elif tag.startswith("I-") and current_entity is not None:
                current_entity.append(word)
            else:
                if current_entity:
                    if current_tag not in global_tag_dict:
                        global_tag_dict[current_tag] = []
                    global_tag_dict[current_tag].append(" ".join(current_entity))
                current_entity = None
                current_tag = None

        if current_entity:
            if current_tag not in global_tag_dict:
                global_tag_dict[current_tag] = []
            global_tag_dict[current_tag].append(" ".join(current_entity))

        ner_data = {
        tag: list(set([entity for entity in entities if (tag == 'B-PER' and entities.count(entity) >= 2) or tag != 'B-PER']))
        for tag, entities in global_tag_dict.items()
        if len([entity for entity in entities if (tag == 'B-PER' and entities.count(entity) >= 2) or tag != 'B-PER']) > 0
    }

    ner_data_file = os.path.join(PROCESSED_FOLDER, 'ner_data.json')
    with open(ner_data_file, 'w') as f:
        json.dump(ner_data, f, indent=2)

    tf_idf_object = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, dtype='float32')
    X_vectors = tf_idf_object.fit_transform(texts)

    optimal_k = 3 
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_vectors)

    cluster_labels = kmeans.labels_
    cluster_to_text = {cluster_id: [] for cluster_id in range(optimal_k)}
    for i, cluster_id in enumerate(cluster_labels):
        cluster_to_text[cluster_id].append(texts[i])

    cluster_data_file = os.path.join(PROCESSED_FOLDER, 'cluster_data.json')
    with open(cluster_data_file, 'w') as f:
        json.dump(cluster_to_text, f, indent=2)

    hierarchy = ['B-PER', 'B-STAT']
    graph_files = []
    for cluster_id, texts_in_cluster in cluster_to_text.items():
        G = nx.DiGraph()
        edge_labels = {}

        for text in texts_in_cluster:
            prev_nodes = None
            for i in range(len(hierarchy)):
                current_entity = hierarchy[i]
                current_nodes = [ent for ent in ner_data.get(current_entity, []) if ent in text]

                if prev_nodes:
                    for prev in prev_nodes:
                        for curr in current_nodes:
                            edge_text = f"{prev} - {curr}"
                            G.add_edge(prev, curr, relation=edge_text)
                            edge_labels[(prev, curr)] = edge_text

                prev_nodes = current_nodes

        elements = []
        for node in G.nodes(data=True):
            elements.append({
                'data': {
                    'id': node[0],
                    'label': node[0],
                    'cluster': int(cluster_id)
                }
            })
        for edge in G.edges(data=True):
            elements.append({
                'data': {
                    'source': edge[0],
                    'target': edge[1],
                    'label': edge[2].get('relation', '')
                }
            })

        file_name = os.path.join(GRAPH_FOLDER, f'graph_cluster_{cluster_id}.json')
        graph_files.append(file_name)
        with open(file_name, 'w') as f:
            json.dump({'elements': elements}, f, indent=2)

    return

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global hmm_ner_tfidf
    hmm_ner_tfidf = HMM(label)
    hmm_ner_tfidf.train(train_sentences, train_tags)
    print("Model HMM telah dimuat dan siap digunakan.")

@app.post("/upload-csv/") 
async def upload_csv(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        df = pd.read_csv(file_path)
        
        if 'Content' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain a 'Content' column")

        df_cleaned = preprocess_and_label(df, hmm_ner_tfidf)
        df_cleaned = process_csv_and_add_descriptions(df_cleaned)
        X_test = vectorizer.transform(df_cleaned['Content_Cleaned']).toarray()
        df_cleaned['Predicted_Label'] = naive_bayes_predict(X_test)
        X_sentiment_test = sentiment_vectorizer.transform(df_cleaned['Content_Cleaned']).toarray()
        df_cleaned['Sentiment'] = naive_bayes_sentiment_predict(X_sentiment_test)

        df_cleaned['Statement_Res'] = df_cleaned['Content_Tokens'].apply(get_hmm2_statement)
        df_cleaned['Combined_Res'] = df_cleaned.apply(
            lambda row: row['Graph_Res'] + row['Statement_Res'] if isinstance(row['Statement_Res'], list) else row['Graph_Res'],
            axis=1
        )


        processed_file_path = os.path.join(PROCESSED_FOLDER, f"{file.filename.split('.')[0]}_processed.csv")
        df_cleaned.to_csv(processed_file_path, index=False)
        process_csv(processed_file_path)

        result = df_cleaned[['Content', 'Predicted_Entities', 'Predicted_Label', 'Statement_Res']].to_dict(orient="records")
        return {"processed_data": result, "processed_file": processed_file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "NER and Classification API is running!"}
