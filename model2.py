import numpy as np 
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from sentence_transformers import SentenceTransformer

orig_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSD9HuN3FucH_wv_5y-mEy3yMWyhE4bmzCyELtya8fKfUCO2znL5df8t7eKguuqVt37UcGw_tEbeT1L/pub?gid=1162126530&single=true&output=csv',on_bad_lines='skip')
orig_df = orig_df.dropna()
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = sbert_model.encode(orig_df['Descripcion'])

query_en = "It is a compulsory insurance for all vehicles that circulate in the national territory that covers bodily harm caused to people in traffic accidents. Foreign vehicles circulating on the country's roads are included. Although I do not know, those vehicles that move on railways and agricultural machinery are taken into account. Compulsory Traffic Accident Insurance, in order to guarantee the resources that facilitate comprehensive care for victims of traffic accidents, in accordance with defined coverage. In this way, all road actors, national or foreign, whether drivers, passengers or pedestrians benefit from this insurance."

#query_vec = sbert_model.encode([query_en])[0]
orig_df['ID']=orig_df['ID'].astype('int')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



def recomend2():
    
    orig_df['c'] = orig_df.apply(lambda row: cosine(sentence_embeddings[7], sentence_embeddings[row['ID']-1]), axis=1)    
    #orig_df['c'] = orig_df.apply(lambda row: cosine(query_vec, sbert_model.encode([row['Descripcion']])[0]), axis=1)

    #orig_df = sorted(orig_df, key=lambda x: x[3], reverse=True)
    orig_df.sort_values(by=['c'])
    return orig_df

print(recomend2())