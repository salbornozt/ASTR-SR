import numpy as np 
import pandas as pd
import pickle
from flask import Flask
from flask import jsonify
from flask import request


orig_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSD9HuN3FucH_wv_5y-mEy3yMWyhE4bmzCyELtya8fKfUCO2znL5df8t7eKguuqVt37UcGw_tEbeT1L/pub?gid=1162126530&single=true&output=csv',on_bad_lines='skip')
orig_df = orig_df.dropna()

#read sentence vectors
with open('my-embeddings.pkl', 'rb') as pkl:
    sentence_embeddings = pickle.load(pkl)
    corpus_sentences = sentence_embeddings['sentences']
    corpus_embeddings = sentence_embeddings['embeddings']


orig_df['ID']=orig_df['ID'].astype('int')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def recomend2(cod_seguro):
    sim_scores =orig_df

    sim_scores['similarity'] = sim_scores.apply(lambda row: cosine(corpus_embeddings[cod_seguro], corpus_embeddings[row['ID']-1]), axis=1)    
    #orig_df.sort_values(by=['c'])
    
    
    sim_scores.sort_values(by=['similarity'], inplace=True, ascending=False)
    #sim_scores = sim_scores[1:4]
    #orig_df.drop([0,1], axis=0, inplace=True)

    #orig_df['json'] = orig_df.apply(lambda x: x.to_json(), axis=1)


    return sim_scores


#print(recomend2(2).to_json(orient='records'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    print(data.get('nom_seg',0))
    codSeguro = data.get('nom_seg',0)
    #prediction = recomend2(data.get('nom_seg',0)).json.values.tolist()
    if codSeguro > 0:
        codSeguro = codSeguro - 1
    prediction = recomend2(codSeguro)
    
    return prediction.to_json(orient='records')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


app.run(port=3000)