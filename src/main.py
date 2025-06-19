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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Reddit parser."""
    logger.info("Starting Reddit Parser")
    
    # Setup databases
    conn = setup_database()
    vector_db = setup_vector_database()
    
    try:
        # Step 1: Fetch posts from Reddit (if needed)
        subreddit_name = "smallbusiness"
        fetch_posts(subreddit_name, limit=100)  # Adjust limit as needed
        
        # Step 2: Populate vector database if empty
        if not check_vector_db_population(vector_db):
            logger.info("Vector database is empty, populating from SQLite...")
            populate_vector_db_from_sqlite(conn, vector_db)
        else:
            logger.info("Vector database already populated, skipping population step.")
        
        # Step 3: Print database summary
        print_database_summary(conn)
        
        # Step 4: Search for business patterns
        logger.info("\nSearching for business patterns...")
        search_business_patterns(vector_db, n_results=10)
        
        # Step 5: Analyze posts with OpenAI
        question = "What are the most common challenges faced by small business owners?"
        analyze_posts(question)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 