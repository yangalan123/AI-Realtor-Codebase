# app.py
from flask import Flask, request, render_template, g
import os
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from utils import get_zillow_data
from tqdm import tqdm

app = Flask(__name__)
app.config['SQLITE_DATABASE'] = 'properties.db'
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')

zillow_text_data, zillow_original_data = get_zillow_data()
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['SQLITE_DATABASE'])
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    db_file = app.config['SQLITE_DATABASE']
    if not os.path.exists(db_file):
        with app.app_context():
            db = get_db()
            with app.open_resource('schema.sql', mode='r') as f:
                db.cursor().executescript(f.read())
            db.commit()
    else:
        print(f"Database {db_file} already exists.")

def load_data():
    with app.app_context():
        db = get_db()
        existing_rows = db.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
        if existing_rows == len(zillow_original_data):
            print("Data already loaded in the database.")
            return

        for listing in tqdm(zillow_original_data):
            listing_id = listing['metadata']['id']
            features = listing['features']
            metadata = listing['metadata']

            description = "{} House features: {}".format(metadata['description'], features['description'])
            # Compute the embedding for the description
            description_embedding = model.encode([description])[0]

            new_listing = {
                'bedrooms': features['Bedrooms'] if features['Bedrooms'] else 1,
                'bathrooms': features['Bathrooms'] if features['Bathrooms'] else 1,
                'home_type': features['Home Type'],
                'year_built': features['Year Built'] if features['Year Built'] else 1900,
                "school_rating": features['school_rating'],
                'avg_school_rating': metadata['avg_school_rating'],
                'price': metadata['price'],
                'street_address': metadata['street_address'],
                'region_string': metadata['regionString'],
                'description': description,
                'embedding': description_embedding.tobytes()  # Convert numpy array to bytes
            }

            db.execute(
                "INSERT INTO properties (id, bedrooms, bathrooms, home_type, year_built, school_rating, avg_school_rating, price, street_address, region_string, description, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (listing_id, new_listing['bedrooms'], new_listing['bathrooms'], new_listing['home_type'], new_listing['year_built'], new_listing['school_rating'], new_listing['avg_school_rating'], new_listing['price'], new_listing['street_address'], new_listing['region_string'], new_listing['description'], new_listing['embedding'])
            )
        db.commit()

def soft_match(query):
    logger = logging.getLogger('soft_match')  # Get a logger specific to this function
    logger.info('Starting soft_match function')

    db = get_db()
    query_embedding = model.encode([query])[0]
    logger.info(f'Query embedding computed. Shape: {query_embedding.shape}')

    # Fetch all property IDs and embeddings from the database
    db_rows = db.execute("SELECT id, embedding FROM properties").fetchall()
    logger.info(f'Fetched {len(db_rows)} rows from database')

    results = []
    for row in db_rows:
        property_id, embedding_blob = row
        property_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(query_embedding, property_embedding)
        results.append((property_id, similarity))
        if (len(results) + 1) % 50 == 0:
            logger.info(f'Computed similarity for {len(results) + 1} properties')

    results.sort(key=lambda x: x[1], reverse=True)
    logger.info('Completed similarity calculations and sorted the results')

    return results

def search(query, hard_features):
    logger = logging.getLogger('search')  # Get a logger specific to this function
    logger.info(f'Starting search with query: {query} and hard_features: {hard_features}')

    db = get_db()
    conditions = [
        f"bedrooms = {hard_features['bedrooms']}",
        f"bathrooms = {hard_features['bathrooms']}",
        f"home_type = '{hard_features['home_type']}'",
        f"year_built = {hard_features['year_built']}",
        f"school_rating = {hard_features['school_rating']}",
        f"price = {hard_features['price']}",
        f"street_address = '{hard_features['street_address']}'",
        f"region_string = '{hard_features['region_string']}'",
        f"avg_school_rating = {hard_features['avg_school_rating']}"
    ]
    logger.info('Constructed conditions for hard feature match')

    hard_match_query = "SELECT id, description, embedding FROM properties WHERE " + " AND ".join(conditions)
    logger.info(f'Executing hard match query: {hard_match_query}')
    hard_match_results = db.execute(hard_match_query).fetchall()
    logger.info(f'Found {len(hard_match_results)} hard match results')

    soft_scores = soft_match(query)
    logger.info('Completed soft matching')

    results = []
    for listing_id, score in soft_scores:
        for result in hard_match_results:
            if result[0] == listing_id:
                embedding = np.frombuffer(result[2], dtype=np.float32)
                results.append((result[:2] + (embedding,), score))
    logger.info(f'Total matched results: {len(results)}')

    return results

@app.route('/', methods=['GET', 'POST'])
def home():
    logger = logging.getLogger('home')  # Get a logger specific to this function
    if request.method == 'POST':
        logger.info('Handling POST request')
        query = request.form['query']
        logger.info(f'Received query: {query}')
        hard_features = {
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'home_type': request.form['home_type'],
            'year_built': int(request.form['year_built']),
            'school_rating': float(request.form['school_rating']),
            'price': int(request.form['price']),
            'street_address': request.form['street_address'],
            'region_string': request.form['region_string'],
            'avg_school_rating': float(request.form['avg_school_rating'])
        }
        logger.info(f'Hard features: {hard_features}')
        results = search(query, hard_features)
        logger.info(f'Found {len(results)} results')
        return render_template('results.html', results=results)

    logger.info('Handling GET request')
    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        init_db()
        load_data()
    app.run(debug=True)