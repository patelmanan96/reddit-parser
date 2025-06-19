# Reddit Parser

A Python script to fetch the top 50 trending posts from any Reddit community.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a Reddit API application:
   - Go to https://www.reddit.com/prefs/apps
   - Click "create another app..."
   - Fill in the required information
   - Select "script" as the application type
   - Set the redirect uri to http://localhost:8080
   - Note down your client ID and client secret

3. Create a `.env` file in the project root with the following content:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=python:reddit_parser:v1.0 (by /u/YourUsername)
```

## Usage

Run the script:
```bash
python reddit_parser.py
```

By default, it will fetch the top 50 posts from r/python. To modify the subreddit or other parameters, edit the `main()` function in `reddit_parser.py`.

## Features

- Fetches top 50 posts from any subreddit
- Displays post title, score, author, creation time, URL, and number of comments
- Configurable time filter (hour, day, week, month, year, all)
- Error handling for API authentication and rate limits 