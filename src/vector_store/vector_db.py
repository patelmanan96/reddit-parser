import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import logging
import sys
import os
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_setup import setup_database, get_all_posts

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_vector_database():
    """Initialize ChromaDB for vector storage and semantic search using a local embedding model."""
    logger.info("Setting up vector database with local embeddings...")
    client = chromadb.Client()

    # Load local sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define a custom embedding function
    class LocalEmbeddingFunction:
        def __call__(self, input):
            return model.encode(input).tolist()
        def name(self):
            return "local-sentence-transformer"

    embedding_function = LocalEmbeddingFunction()

    # Create or get the collection
    collection = client.get_or_create_collection(
        name="reddit_posts",
        embedding_function=embedding_function
    )
    return collection

def check_vector_db_population(vector_db):
    """Check if vector database is already populated."""
    try:
        # Try to get count of documents
        count = vector_db.count()
        return count > 0
    except Exception:
        return False

def populate_vector_db_from_sqlite(conn, vector_db):
    """Populate vector database with existing posts from SQLite."""
    logger.info("Populating vector database from SQLite...")
    cursor = conn.cursor()
    
    # Get all posts from SQLite
    cursor.execute("SELECT id, title, selftext, score, author, created_utc, num_comments FROM posts")
    posts = cursor.fetchall()
    
    # Process posts in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(posts) + batch_size - 1)//batch_size}")
        
        for post in batch:
            post_id, title, selftext, score, author, created_utc, num_comments = post
            # Combine title and content for better semantic search
            content = f"Title: {title}\nContent: {selftext}"
            
            # Add to vector database
            vector_db.add(
                documents=[content],
                metadatas=[{
                    'post_id': post_id,
                    'title': title,
                    'score': score,
                    'author': author,
                    'created_utc': created_utc,
                    'num_comments': num_comments
                }],
                ids=[post_id]
            )
    
    logger.info(f"Vector database populated with {len(posts)} posts")

def find_similar_posts(collection, query, n_results=5):
    """Find similar posts using semantic search."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def search_business_patterns(vector_db, pattern_type=None, n_results=10, return_json=False):
    """Search for specific business patterns in the posts."""
    # Define pattern-specific queries
    pattern_queries = {
        'revenue': [
            "monthly revenue growth",
            "revenue streams",
            "profit margins",
            "income sources"
        ],
        'marketing': [
            "marketing strategies",
            "customer acquisition",
            "advertising methods",
            "social media marketing"
        ],
        'scaling': [
            "business growth",
            "scaling operations",
            "hiring employees",
            "expanding business"
        ],
        'challenges': [
            "business challenges",
            "common problems",
            "difficulties faced",
            "obstacles overcome"
        ],
        'success': [
            "successful strategies",
            "what worked",
            "key to success",
            "winning approaches"
        ]
    }
    
    # If no specific pattern type is provided, search across all patterns
    if pattern_type:
        queries = pattern_queries.get(pattern_type, [])
    else:
        queries = [q for pattern in pattern_queries.values() for q in pattern]
    
    results = []
    seen_ids = set()  # To avoid duplicates
    
    for query in queries:
        search = find_similar_posts(vector_db, query, n_results=n_results)
        for doc, meta in zip(search['documents'][0], search['metadatas'][0]):
            if meta['post_id'] not in seen_ids:
                results.append({
                    'title': meta['title'],
                    'content': doc,
                    'score': meta['score'],
                    'comments': meta['num_comments'],
                    'pattern_type': pattern_type if pattern_type else 'general',
                    'query': query
                })
                seen_ids.add(meta['post_id'])
    
    # Sort by score and comments
    results = sorted(results, key=lambda x: (x['score'], x['comments']), reverse=True)
    
    if return_json:
        return results[:n_results]
    else:
        # Print results
        print(f"\nBusiness Pattern Analysis Results ({pattern_type if pattern_type else 'All Patterns'}):")
        print("-" * 50)
        for i, r in enumerate(results[:n_results], 1):
            print(f"\nResult {i}:")
            print(f"Title: {r['title']}")
            print(f"Pattern Type: {r['pattern_type']}")
            print(f"Matched Query: {r['query']}")
            print(f"Score: {r['score']}, Comments: {r['comments']}")
            print(f"Content Preview: {r['content'][:200]}...")
            print("-" * 50)
        return results[:n_results]

if __name__ == "__main__":
    # Test vector database setup and population
    conn = setup_database()
    vector_db = setup_vector_database()
    
    if not check_vector_db_population(vector_db):
        populate_vector_db_from_sqlite(conn, vector_db)
    
    # Test search functionality
    search_business_patterns(vector_db, pattern_type='revenue', n_results=5)
    
    conn.close() 