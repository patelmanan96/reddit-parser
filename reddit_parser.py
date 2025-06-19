import os
import praw
from dotenv import load_dotenv
from datetime import datetime, timedelta
import openai
from typing import List, Dict, Any
import time
import json
from pathlib import Path
import logging
import sqlite3
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for rate limiting and caching
CACHE_DIR = Path("cache")
CACHE_EXPIRY = timedelta(hours=24)  # Increased cache expiry
REDDIT_RATE_LIMIT_DELAY = 0.5  # Reduced delay for faster fetching
MAX_POSTS_PER_REQUEST = 100  # Reduced to avoid rate limits
MAX_COMMENTS_PER_POST = 10
BATCH_SIZE = 10  # Number of posts to analyze in one GPT call
DB_PATH = "reddit_analysis.db"
TOTAL_POSTS_TO_FETCH = 10000  # New constant for total posts to fetch
SAVE_INTERVAL = 100  # Save to cache every 100 posts

def setup_reddit_client():
    """Initialize and return a Reddit client instance."""
    logger.info("Setting up Reddit client...")
    load_dotenv()
    
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'python:reddit_parser:v1.0 (by /u/YourUsername)')
    
    if not all([client_id, client_secret]):
        raise ValueError("Missing Reddit API credentials. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file")
    
    logger.info("Reddit client setup complete")
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

def setup_openai_client():
    """Initialize OpenAI API key."""
    logger.info("Setting up OpenAI client...")
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Missing OpenAI API key. Please set OPENAI_API_KEY in .env file")
    openai.api_key = api_key
    logger.info("OpenAI client setup complete")

def get_cached_data(cache_key: str) -> Dict:
    """Get data from cache if it exists and is not expired."""
    logger.info(f"Checking cache for key: {cache_key}")
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time < CACHE_EXPIRY:
                logger.info("Found valid cached data")
                return data['content']
    logger.info("No valid cache found")
    return None

def save_to_cache(cache_key: str, content: Dict):
    """Save data to cache with timestamp."""
    logger.info(f"Saving data to cache with key: {cache_key}")
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'content': content
    }
    
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    logger.info("Cache save complete")

def get_cached_progress(cache_key: str):
    """Get pagination progress from cache if it exists and is not expired."""
    logger.info(f"Checking progress cache for key: {cache_key}")
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}_progress.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time < CACHE_EXPIRY:
                logger.info("Found valid progress cache")
                return data['progress']
    logger.info("No valid progress cache found")
    return None

def save_progress_to_cache(cache_key: str, progress: dict):
    """Save pagination progress to cache with timestamp."""
    logger.info(f"Saving progress to cache with key: {cache_key}")
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}_progress.json"
    data = {
        'timestamp': datetime.now().isoformat(),
        'progress': progress
    }
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    logger.info("Progress cache save complete")

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

    # Create comments table if it doesn't exist
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

def save_post_to_db(conn, post_data: Dict):
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

def save_analysis_to_db(conn, post_id: str, question: str, answer: str):
    """Save analysis results to the database."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO analysis_results (post_id, question, answer, created_at)
    VALUES (?, ?, ?, ?)
    ''', (post_id, question, datetime.now().isoformat(), answer))
    conn.commit()

def save_comment_to_db(conn, comment_data: dict):
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

def save_post_to_vector_db(collection, post_data: Dict):
    """Save a post's content to the vector database."""
    # Combine title and content for better semantic search
    content = f"Title: {post_data['title']}\nContent: {post_data['selftext']}"
    
    # Add to vector database
    collection.add(
        documents=[content],
        metadatas=[{
            'post_id': post_data['id'],
            'title': post_data['title'],
            'score': post_data['score'],
            'author': post_data['author'],
            'created_utc': post_data['created_utc'],
            'num_comments': post_data['num_comments']
        }],
        ids=[post_data['id']]
    )

def find_similar_posts(collection, query: str, n_results: int = 5):
    """Find similar posts using semantic search."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def analyze_post_similarity(collection, post_id: str, n_results: int = 5):
    """Find posts similar to a specific post."""
    # Get the post's content from the collection
    post = collection.get(ids=[post_id])
    if not post['documents']:
        return None
    
    # Find similar posts
    results = collection.query(
        query_texts=[post['documents'][0]],
        n_results=n_results + 1  # +1 because the post itself will be included
    )
    
    # Filter out the original post
    filtered_results = {
        'documents': [],
        'metadatas': [],
        'distances': []
    }
    
    for i, metadata in enumerate(results['metadatas'][0]):
        if metadata['post_id'] != post_id:
            filtered_results['documents'].append(results['documents'][0][i])
            filtered_results['metadatas'].append(metadata)
            filtered_results['distances'].append(results['distances'][0][i])
    
    return filtered_results

def get_posts(subreddit_name, limit=TOTAL_POSTS_TO_FETCH, time_filter='all', sort_by='top'):
    """Fetch posts with improved pagination support and fetch top 10 comments per post. On error, save progress, wait, and resume."""
    logger.info(f"Fetching posts from r/{subreddit_name}")
    cache_key = f"{subreddit_name}_{sort_by}_{time_filter}_{limit}_no_comments"
    progress = get_cached_progress(cache_key)
    
    post_list = []
    already_fetched = set()
    after = None
    if progress:
        post_list = progress.get('post_list', [])
        already_fetched = set(progress.get('already_fetched', []))
        after = progress.get('after')
        logger.info(f"Resuming from {len(post_list)} posts already fetched")
    else:
        logger.info("No progress cache found, starting fresh")
    
    reddit = setup_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)
    
    retry_count = 0
    max_retries = 3
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    with tqdm(total=limit, initial=len(post_list), desc="Fetching posts") as pbar:
        while len(post_list) < limit:
            try:
                if sort_by == 'new':
                    posts = subreddit.new(limit=min(MAX_POSTS_PER_REQUEST, limit - len(post_list)), params={'after': after})
                elif sort_by == 'hot':
                    posts = subreddit.hot(limit=min(MAX_POSTS_PER_REQUEST, limit - len(post_list)), params={'after': after})
                elif sort_by == 'top':
                    posts = subreddit.top(time_filter=time_filter, limit=min(MAX_POSTS_PER_REQUEST, limit - len(post_list)), params={'after': after})
                elif sort_by == 'rising':
                    posts = subreddit.rising(limit=min(MAX_POSTS_PER_REQUEST, limit - len(post_list)), params={'after': after})
                else:
                    raise ValueError("Invalid sort method")
                
                batch = []
                last_post_fullname = None
                for post in posts:
                    if post.id in already_fetched:
                        continue
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'score': post.score,
                        'author': str(post.author),
                        'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'url': post.url,
                        'num_comments': post.num_comments,
                        'permalink': f"https://reddit.com{post.permalink}",
                        'selftext': post.selftext,
                        'is_self': post.is_self,
                        'link_flair_text': post.link_flair_text
                    }
                    batch.append(post_data)
                    pbar.update(1)
                    already_fetched.add(post.id)
                    last_post_fullname = f"t3_{post.id}"
                    # Fetch top 10 comments for this post
                    try:
                        post.comments.replace_more(limit=0)
                        top_comments = sorted(post.comments.list(), key=lambda c: getattr(c, 'score', 0), reverse=True)[:10]
                        post_data['comments'] = []
                        for comment in top_comments:
                            if hasattr(comment, 'body'):
                                comment_data = {
                                    'id': comment.id,
                                    'post_id': post.id,
                                    'author': str(comment.author),
                                    'score': getattr(comment, 'score', 0),
                                    'created_utc': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                    'body': comment.body
                                }
                                post_data['comments'].append(comment_data)
                    except Exception as e:
                        logger.warning(f"Failed to fetch comments for post {post.id}: {e}")
                        post_data['comments'] = []
                
                if not batch:
                    logger.warning("No more posts available")
                    break
                
                post_list.extend(batch)
                after = last_post_fullname
                
                # Save progress to cache
                if len(post_list) % SAVE_INTERVAL == 0:
                    save_progress_to_cache(cache_key, {
                        'post_list': post_list,
                        'already_fetched': list(already_fetched),
                        'after': after
                    })
                    logger.info(f"Saved progress: {len(post_list)} posts, after={after}")
                
                # Reset error counters on successful fetch
                retry_count = 0
                consecutive_errors = 0
                time.sleep(REDDIT_RATE_LIMIT_DELAY)
            except Exception as e:
                retry_count += 1
                consecutive_errors += 1
                logger.error(f"Error fetching posts (attempt {retry_count}/{max_retries}): {str(e)}")
                save_progress_to_cache(cache_key, {
                    'post_list': post_list,
                    'already_fetched': list(already_fetched),
                    'after': after
                })
                logger.info(f"Saved progress at {len(post_list)} posts. Waiting 60 seconds before retry...")
                time.sleep(60)
                if retry_count >= max_retries or consecutive_errors >= max_consecutive_errors:
                    logger.error("Max retries or consecutive errors reached, saving progress and stopping")
                    break
                sleep_time = REDDIT_RATE_LIMIT_DELAY * (2 ** retry_count)
                logger.info(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
    logger.info(f"Fetched {len(post_list)} posts")
    save_progress_to_cache(cache_key, {
        'post_list': post_list,
        'already_fetched': list(already_fetched),
        'after': after
    })
    return post_list

def analyze_content_with_llm(posts: List[Dict[str, Any]], question: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    """Analyze Reddit content using OpenAI's GPT model with batching."""
    logger.info("Starting LLM analysis")
    if not dry_run:
        setup_openai_client()
    
    results = []
    
    # Process posts in batches
    for i in range(0, len(posts), BATCH_SIZE):
        batch = posts[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(posts) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        # Prepare context for the batch
        context = []
        for post in batch:
            post_context = f"Title: {post['title']}\n"
            if post['is_self'] and post['selftext']:
                post_context += f"Content: {post['selftext']}\n"
            post_context += f"Category: {post['link_flair_text']}\n"
            post_context += f"Score: {post['score']}\n"
            post_context += f"Comments: {post['num_comments']}\n"
            context.append(post_context)
        
        if dry_run:
            # Simulate GPT response for dry run
            mock_response = f"[DRY RUN] Would analyze {len(batch)} posts for question: {question[:50]}..."
            for post in batch:
                results.append({
                    'post_id': post['id'],
                    'question': question,
                    'answer': mock_response
                })
            time.sleep(0.1)  # Small delay to simulate processing
            continue
            
        full_context = "\n---\n".join(context)
        
        prompt = f"""Based on the following Reddit posts from r/smallbusiness, please answer this question:
{question}

Context:
{full_context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Cites specific examples from the posts
3. Provides insights and patterns observed in the community
4. Offers practical advice or solutions if applicable
5. Include specific revenue numbers or success metrics when mentioned
6. Analyze market trends and opportunities
7. Identify potential risks and challenges

Answer:"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business analyst assistant that analyzes Reddit content to provide insights about small business trends, opportunities, and challenges."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Store results with post IDs for database storage
            for post in batch:
                results.append({
                    'post_id': post['id'],
                    'question': question,
                    'answer': response.choices[0].message.content
                })
            
            # Rate limiting
            time.sleep(1)
            
        except openai.RateLimitError as e:
            error_msg = """OpenAI API quota exceeded. Please take one of the following actions:
1. Check your OpenAI API usage and billing details at https://platform.openai.com/account/usage
2. Upgrade your plan if needed at https://platform.openai.com/account/billing
3. Wait until your quota resets in the next billing period

For more information, visit: https://platform.openai.com/docs/guides/error-codes/api-errors"""
            logger.error(f"OpenAI API quota exceeded: {str(e)}")
            return results
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return results
    
    return results

def print_database_summary(conn):
    """Print a summary of the database contents."""
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
    
    print("\nDatabase Summary:")
    print("-" * 50)
    print(f"Total Posts: {post_count}")
    print(f"Total Analysis Results: {analysis_count}")
    print("\nQuestions Analyzed:")
    for q in questions:
        print(f"- {q[0][:100]}...")
    print("-" * 50)

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

def extract_revenue(text):
    # Simple regex to find $ amounts (e.g., $10,000 or $10k)
    matches = re.findall(r'\$([\d,]+(?:\.\d+)?)([kKmM]?)', text)
    max_revenue = 0
    for amount, suffix in matches:
        amount = float(amount.replace(',', ''))
        if suffix.lower() == 'k':
            amount *= 1_000
        elif suffix.lower() == 'm':
            amount *= 1_000_000
        if amount > max_revenue:
            max_revenue = amount
    return max_revenue

def find_most_profitable_business(vector_db, n_results=20):
    # Search for posts likely to mention profit/revenue
    queries = [
        "monthly revenue",
        "profit per month",
        "business making the most money",
        "highest revenue",
        "most profitable business"
    ]
    results = []
    for query in queries:
        search = find_similar_posts(vector_db, query, n_results=n_results)
        for doc, meta in zip(search['documents'][0], search['metadatas'][0]):
            revenue = extract_revenue(doc)
            if revenue > 0:
                results.append({
                    'title': meta['title'],
                    'revenue': revenue,
                    'content': doc[:300] + '...',
                    'score': meta['score'],
                    'comments': meta['num_comments']
                })
    # Sort by revenue
    results = sorted(results, key=lambda x: x['revenue'], reverse=True)
    # Print top 5
    print("\nMost Profitable Businesses Found:")
    print("-" * 50)
    for r in results[:5]:
        print(f"Title: {r['title']}")
        print(f"Estimated Revenue: ${r['revenue']:,.0f}")
        print(f"Score: {r['score']}, Comments: {r['comments']}")
        print(f"Content Preview: {r['content']}")
        print("-" * 50)

def search_business_patterns(vector_db, pattern_type=None, n_results=10):
    """Search for specific business patterns in the posts.
    
    Args:
        vector_db: The vector database instance
        pattern_type: Optional specific pattern to search for (e.g., 'revenue', 'marketing', 'scaling')
        n_results: Number of results to return
    """
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
    
    return results

def check_vector_db_population(vector_db):
    """Check if vector database is already populated."""
    try:
        # Try to get count of documents
        count = vector_db.count()
        return count > 0
    except Exception:
        return False

def analyze_high_revenue_opportunities(vector_db, target_revenue=50_000_000, n_results=20):
    """Analyze Reddit posts to identify business opportunities with high revenue potential.
    
    Args:
        vector_db: The vector database instance
        target_revenue: Target annual revenue in dollars
        n_results: Number of results to return
    """
    # Define queries to find high-revenue business opportunities
    queries = [
        "successful business with high profit margins",
        "business making millions in revenue",
        "scalable business model with high margins",
        "business with low overhead and high revenue",
        "recurring revenue business model",
        "subscription based business success",
        "business with high profit margins",
        "business with low competition and high demand",
        "business with high customer lifetime value",
        "business with high profit per customer"
    ]
    
    results = []
    seen_ids = set()
    
    for query in queries:
        search = find_similar_posts(vector_db, query, n_results=n_results)
        for doc, meta in zip(search['documents'][0], search['metadatas'][0]):
            if meta['post_id'] not in seen_ids:
                # Extract revenue information
                revenue = extract_revenue(doc)
                if revenue > 0:
                    results.append({
                        'title': meta['title'],
                        'content': doc,
                        'revenue': revenue,
                        'score': meta['score'],
                        'comments': meta['num_comments'],
                        'query': query
                    })
                seen_ids.add(meta['post_id'])
    
    # Sort by revenue
    results = sorted(results, key=lambda x: x['revenue'], reverse=True)
    
    # Print results
    print(f"\nHigh Revenue Business Opportunities Analysis:")
    print("-" * 50)
    print(f"Target Annual Revenue: ${target_revenue:,.0f}")
    print("-" * 50)
    
    for i, r in enumerate(results[:n_results], 1):
        print(f"\nOpportunity {i}:")
        print(f"Title: {r['title']}")
        print(f"Estimated Revenue: ${r['revenue']:,.0f}")
        print(f"Score: {r['score']}, Comments: {r['comments']}")
        print(f"Matched Query: {r['query']}")
        print(f"Content Preview: {r['content'][:200]}...")
        print("-" * 50)
    
    return results

def main():
    logger.info("Starting Reddit Parser")
    conn = setup_database()
    vector_db = setup_vector_database()
    
    try:
        # Only populate vector database if it's empty
        if not check_vector_db_population(vector_db):
            logger.info("Vector database is empty, populating from SQLite...")
            populate_vector_db_from_sqlite(conn, vector_db)
        else:
            logger.info("Vector database already populated, skipping population step.")
        
        # Print database summary
        print_database_summary(conn)
        
        # Analyze high revenue business opportunities
        logger.info("\nAnalyzing high revenue business opportunities...")
        analyze_high_revenue_opportunities(vector_db, target_revenue=50_000_000)
        
        # Search for business patterns
        logger.info("\nSearching for business patterns...")
        search_business_patterns(vector_db, n_results=10)
        
        # Find and print most profitable businesses
        find_most_profitable_business(vector_db)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
