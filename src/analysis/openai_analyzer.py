import os
import openai
from dotenv import load_dotenv
import logging
import time
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_setup import setup_database, save_analysis_to_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 10  # Number of posts to analyze in one GPT call

def setup_openai_client():
    """Initialize OpenAI API key."""
    logger.info("Setting up OpenAI client...")
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Missing OpenAI API key. Please set OPENAI_API_KEY in .env file")
    openai.api_key = api_key
    logger.info("OpenAI client setup complete")

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

def analyze_posts(question: str, dry_run: bool = False, return_json: bool = False):
    """Analyze posts from the database using OpenAI."""
    conn = setup_database()
    cursor = conn.cursor()
    
    # Get posts that haven't been analyzed for this question
    cursor.execute("""
    SELECT p.* FROM posts p
    LEFT JOIN analysis_results ar ON p.id = ar.post_id AND ar.question = ?
    WHERE ar.id IS NULL
    """, (question,))
    
    posts = []
    for row in cursor.fetchall():
        posts.append({
            'id': row[0],
            'title': row[1],
            'score': row[2],
            'author': row[3],
            'created_utc': row[4],
            'url': row[5],
            'num_comments': row[6],
            'permalink': row[7],
            'selftext': row[8],
            'is_self': row[9],
            'link_flair_text': row[10]
        })
    
    if not posts:
        logger.info("No new posts to analyze")
        result = []
        if return_json:
            conn.close()
            return result
        else:
            conn.close()
            return
    
    # Analyze posts
    results = analyze_content_with_llm(posts, question, dry_run)
    
    # Save results to database
    for result in results:
        save_analysis_to_db(conn, result['post_id'], result['question'], result['answer'])
    
    conn.close()
    if return_json:
        return results
    else:
        return results

if __name__ == "__main__":
    # Test OpenAI analysis
    question = "What are the most common challenges faced by small business owners?"
    analyze_posts(question, dry_run=True)  # Use dry_run=True for testing 