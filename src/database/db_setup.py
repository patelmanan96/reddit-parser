import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = "reddit_analysis.db"

def setup_database():
    """Initialize SQLite database for storing analysis results."""
    logger.info("Setting up database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        title TEXT,
        score INTEGER,
        author TEXT,
        created_utc TEXT,
        url TEXT,
        num_comments INTEGER,
        permalink TEXT,
        selftext TEXT,
        is_self BOOLEAN,
        link_flair_text TEXT,
        analyzed BOOLEAN DEFAULT 0
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id TEXT,
        question TEXT,
        answer TEXT,
        created_at TEXT,
        FOREIGN KEY (post_id) REFERENCES posts (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id TEXT PRIMARY KEY,
        post_id TEXT,
        author TEXT,
        score INTEGER,
        created_utc TEXT,
        body TEXT,
        FOREIGN KEY (post_id) REFERENCES posts (id)
    )
    ''')
    
    conn.commit()
    return conn

def save_post_to_db(conn, post_data):
    """Save a post to the database if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR IGNORE INTO posts 
    (id, title, score, author, created_utc, url, num_comments, permalink, selftext, is_self, link_flair_text)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        post_data['id'],
        post_data['title'],
        post_data['score'],
        post_data['author'],
        post_data['created_utc'],
        post_data['url'],
        post_data['num_comments'],
        post_data['permalink'],
        post_data['selftext'],
        post_data['is_self'],
        post_data['link_flair_text']
    ))
    conn.commit()

def save_comment_to_db(conn, comment_data):
    """Save a comment to the database if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR IGNORE INTO comments (id, post_id, author, score, created_utc, body)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        comment_data['id'],
        comment_data['post_id'],
        comment_data['author'],
        comment_data['score'],
        comment_data['created_utc'],
        comment_data['body']
    ))
    conn.commit()

def save_analysis_to_db(conn, post_id, question, answer):
    """Save analysis results to the database."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO analysis_results (post_id, question, answer, created_at)
    VALUES (?, ?, ?, ?)
    ''', (post_id, question, datetime.now().isoformat(), answer))
    conn.commit()

def get_all_posts(conn):
    """Get all posts from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts")
    return cursor.fetchall()

def print_database_summary(conn, return_json=False):
    """Print or return a summary of the database contents."""
    cursor = conn.cursor()
    
    # Get post count
    cursor.execute("SELECT COUNT(*) FROM posts")
    post_count = cursor.fetchone()[0]
    
    # Get analysis count
    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    analysis_count = cursor.fetchone()[0]
    
    # Get unique questions
    cursor.execute("SELECT DISTINCT question FROM analysis_results")
    questions = cursor.fetchall()
    
    if return_json:
        return {
            'total_posts': post_count,
            'total_analysis_results': analysis_count,
            'questions_analyzed': [q[0] for q in questions]
        }
    else:
        print("\nDatabase Summary:")
        print("-" * 50)
        print(f"Total Posts: {post_count}")
        print(f"Total Analysis Results: {analysis_count}")
        print("\nQuestions Analyzed:")
        for q in questions:
            print(f"- {q[0][:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    # Test database setup
    conn = setup_database()
    print_database_summary(conn)
    conn.close() 