import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_setup import setup_database
from vector_store.vector_db import setup_vector_database, check_vector_db_population, populate_vector_db_from_sqlite, search_business_patterns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_business_metrics(vector_db):
    """Analyze business metrics including revenue, profit, and success factors."""
    print("\n=== Business Revenue Analysis ===")
    search_business_patterns(vector_db, pattern_type='revenue', n_results=10)
    
    print("\n=== Business Profit Analysis ===")
    # Custom profit-focused queries
    profit_queries = [
        "profit margins",
        "net profit",
        "profitability",
        "profit per month",
        "profit growth"
    ]
    for query in profit_queries:
        print(f"\nAnalyzing: {query}")
        results = vector_db.query(
            query_texts=[query],
            n_results=5
        )
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            print(f"\nTitle: {meta['title']}")
            print(f"Score: {meta['score']}, Comments: {meta['num_comments']}")
            print(f"Content Preview: {doc[:200]}...")
            print("-" * 50)

def analyze_best_businesses(vector_db):
    """Analyze what makes a business successful and identify best business types."""
    print("\n=== Best Business Types Analysis ===")
    success_queries = [
        "most successful business",
        "best business to start",
        "highest profit margin business",
        "easiest business to scale",
        "most profitable business model"
    ]
    
    for query in success_queries:
        print(f"\nAnalyzing: {query}")
        results = vector_db.query(
            query_texts=[query],
            n_results=5
        )
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            print(f"\nTitle: {meta['title']}")
            print(f"Score: {meta['score']}, Comments: {meta['num_comments']}")
            print(f"Content Preview: {doc[:200]}...")
            print("-" * 50)

def analyze_business_challenges(vector_db):
    """Analyze common business challenges and how to overcome them."""
    print("\n=== Business Challenges Analysis ===")
    search_business_patterns(vector_db, pattern_type='challenges', n_results=10)

def comprehensive_business_pattern_analysis(vector_db, n_results=10):
    """Replicate the multi-pattern business pattern search from reddit_parser.py."""
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
    results = []
    seen_ids = set()
    for pattern_type, queries in pattern_queries.items():
        for query in queries:
            search = vector_db.query(query_texts=[query], n_results=n_results)
            for doc, meta in zip(search['documents'][0], search['metadatas'][0]):
                if meta['post_id'] not in seen_ids:
                    results.append({
                        'title': meta['title'],
                        'content': doc,
                        'score': meta['score'],
                        'comments': meta['num_comments'],
                        'pattern_type': pattern_type,
                        'query': query
                    })
                    seen_ids.add(meta['post_id'])
    # Sort by score and comments
    results = sorted(results, key=lambda x: (x['score'], x['comments']), reverse=True)
    print(f"\nComprehensive Business Pattern Analysis Results (All Patterns):")
    print("-" * 50)
    for i, r in enumerate(results[:n_results], 1):
        print(f"\nResult {i}:")
        print(f"Title: {r['title']}")
        print(f"Pattern Type: {r['pattern_type']}")
        print(f"Matched Query: {r['query']}")
        print(f"Score: {r['score']}, Comments: {r['comments']}")
        print(f"Content Preview: {r['content'][:200]}...")
        print("-" * 50)
    return results

def recommend_best_niche_business(vector_db, n_results=7):
    """Find the best business to start for high profit, quick returns, and niche appeal."""
    queries = [
        "most profitable niche business",
        "businesses with fastest return on investment",
        "unique business ideas with high income",
        "businesses with low competition and high profit",
        "best business to start for quick money",
        "niche business with high margins",
        "businesses that make money fast in a niche field"
    ]
    results = []
    seen_ids = set()
    for query in queries:
        search = vector_db.query(query_texts=[query], n_results=n_results)
        for doc, meta in zip(search['documents'][0], search['metadatas'][0]):
            if meta['post_id'] not in seen_ids:
                results.append({
                    'title': meta['title'],
                    'content': doc,
                    'score': meta['score'],
                    'comments': meta['num_comments'],
                    'query': query
                })
                seen_ids.add(meta['post_id'])
    # Sort by score and comments
    results = sorted(results, key=lambda x: (x['score'], x['comments']), reverse=True)
    print(f"\nBest Niche Business Recommendations (High Profit, Fast, Niche):")
    print("-" * 50)
    for i, r in enumerate(results[:n_results], 1):
        print(f"\nResult {i}:")
        print(f"Title: {r['title']}")
        print(f"Matched Query: {r['query']}")
        print(f"Score: {r['score']}, Comments: {r['comments']}")
        print(f"Content Preview: {r['content'][:300]}...")
        print("-" * 50)
    return results

def main():
    """Run deep business analysis."""
    logger.info("Starting Deep Business Analysis")
    
    # Setup databases
    conn = setup_database()
    vector_db = setup_vector_database()
    
    try:
        # Ensure vector database is populated
        if not check_vector_db_population(vector_db):
            logger.info("Vector database is empty, populating from SQLite...")
            populate_vector_db_from_sqlite(conn, vector_db)
        else:
            logger.info("Vector database already populated, proceeding with analysis.")
        
        # Run comprehensive business pattern analysis (replicating reddit_parser)
        comprehensive_business_pattern_analysis(vector_db, n_results=10)
        # Recommend best niche business for high profit and quick returns
        recommend_best_niche_business(vector_db, n_results=7)
        # Run comprehensive business analysis
        analyze_business_metrics(vector_db)
        analyze_best_businesses(vector_db)
        analyze_business_challenges(vector_db)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 