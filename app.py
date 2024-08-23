from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Carga de los archivos CSV
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Vectorizar los géneros de las películas
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')  # Maneja valores nulos
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calcular la similitud del coseno entre las películas
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función para obtener recomendaciones
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in movies['title'].values:
        return ["La película no se encuentra en el dataset."]
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Obtener las 10 más similares
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    recommendations = get_recommendations(title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
