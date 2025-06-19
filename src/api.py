from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_setup import setup_database, print_database_summary
from reddit.reddit_fetcher import fetch_posts
from vector_store.vector_db import setup_vector_database, check_vector_db_population, populate_vector_db_from_sqlite, search_business_patterns
from analysis.openai_analyzer import analyze_posts

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/fetch_posts', methods=['POST'])
def api_fetch_posts():
    data = request.json
    subreddit_name = data.get('subreddit', 'smallbusiness')
    limit = int(data.get('limit', 100))
    fetch_posts(subreddit_name, limit=limit)
    return jsonify({'status': 'success', 'message': f'Fetched {limit} posts from r/{subreddit_name}.'})

@app.route('/populate_vector_db', methods=['POST'])
def api_populate_vector_db():
    conn = setup_database()
    vector_db = setup_vector_database()
    if not check_vector_db_population(vector_db):
        populate_vector_db_from_sqlite(conn, vector_db)
        conn.close()
        return jsonify({'status': 'success', 'message': 'Vector DB populated from SQLite.'})
    else:
        conn.close()
        return jsonify({'status': 'skipped', 'message': 'Vector DB already populated.'})

@app.route('/db_summary', methods=['GET'])
def api_db_summary():
    conn = setup_database()
    summary = print_database_summary(conn, return_json=True)
    conn.close()
    return jsonify(summary)

@app.route('/search_patterns', methods=['GET'])
def api_search_patterns():
    n_results = int(request.args.get('n_results', 10))
    vector_db = setup_vector_database()
    results = search_business_patterns(vector_db, n_results=n_results, return_json=True)
    return jsonify({'results': results})

@app.route('/analyze', methods=['POST'])
def api_analyze():
    data = request.json
    question = data.get('question', 'What are the most common challenges faced by small business owners?')
    analysis = analyze_posts(question, return_json=True)
    return jsonify({'analysis': analysis})

if __name__ == '__main__':
    app.run(debug=True) 