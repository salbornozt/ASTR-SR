import numpy as np 
import pandas as pd
from joblib import load
from flask import Flask
from flask import jsonify
from flask import request


orig_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSD9HuN3FucH_wv_5y-mEy3yMWyhE4bmzCyELtya8fKfUCO2znL5df8t7eKguuqVt37UcGw_tEbeT1L/pub?gid=1104602369&single=true&output=csv',on_bad_lines='skip')
orig_df.replace(np.nan, 0)
clf = load('coisne.joblib') 
indices = pd.Series(orig_df.index, index=orig_df['Ramo']).drop_duplicates()


# Function that takes in movie title as input and gives recommendations 
def content_recommender(seguro, cosine_sim=clf, df=orig_df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[seguro]
    idx
    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx + 1]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:4]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['Ramo'].iloc[movie_indices]





app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    print(data.get('nom_seg','Hogar'))
    prediction = content_recommender(data.get('nom_seg','Hogar')).values.tolist()
    return jsonify({
        'prediction' : prediction
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


app.run(port=3000)

