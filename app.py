from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np # Fixed: standard alias is np
from sklearn.metrics.pairwise import cosine_similarity

# loading data
df = pickle.load(open('model/df.pkl','rb'))
tfidf_matrix = pickle.load(open('model/tfidf_matrix.pkl','rb'))
indices = pickle.load(open('model/indices.pkl','rb'))
df_popular = pickle.load(open('model/df_popular.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'index.html',
        books_names = df_popular['title'].tolist(),
        authors = df_popular['authors'].tolist(),
        images = df_popular['thumbnail'].tolist(),
        votes = df_popular['ratings_count'].tolist(),
        ratings = df_popular['average_rating'].tolist(),
        genres = df_popular['categories'].tolist()
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if user_input not in indices:
        return render_template('recommend.html', data=None, error='Book not found.')
    
    try:
        idx = indices[user_input]
        sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        similar_idx = sim_score.argsort()[::-1][1:11]
        
        recommended_books = df.iloc[similar_idx]

        data = []
        for i in range(len(recommended_books)):
            temp_df = recommended_books.iloc[i]
            
            item = [
                temp_df['title'] if 'title' in temp_df else "Unknown",
                temp_df['authors'] if 'authors' in temp_df else "Unknown Author",
                temp_df['categories'] if 'categories' in temp_df else "N/A",
                temp_df['average_rating'] if 'average_rating' in temp_df else 0,
                temp_df['ratings_count'] if 'ratings_count' in temp_df else 0,
                temp_df['thumbnail'] if 'thumbnail' in temp_df else ""
            ]
            data.append(item)

        return render_template('recommend.html', data=data)

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return render_template('recommend.html', data=None, error="An error occurred.")

if __name__ == '__main__':
    app.run(debug=True)