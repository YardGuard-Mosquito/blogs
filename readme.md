# AutoPublisher 🤖📝

Automated article generation and publishing system using Google Gemini AI, Google Sheets, and WordPress

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/yourusername/autopublisher/publish.yml?style=flat-square)](https://github.com/yourusername/autopublisher/actions)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features ✨

- **AI-Powered Content Generation** using Google Gemini
- **Dynamic Topic Management** via Google Sheets integration
- **Automated WordPress Publishing** with Markdown support
- **SEO Optimization** with automatic keyword insertion
- **Reference Management** with URL validation
- **Robust Error Handling** and retry mechanisms
- **Scheduled Publishing** via GitHub Actions
- **CSV Validation** with malformed data detection

## Requirements 📋

- Python 3.10+
- Google Gemini API Key
- WordPress instance with REST API access
- Public Google Sheet with article topics

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autopublisher.git
cd autopublisher

Install dependencies:
python -m pip install -r requirements.txt

Create .env file from template:
cp .env.example .env

Configuration ⚙️

Set these environment variables in your .env file:

Variable	Description	Example
GEMINI_API_KEY	Google Gemini API key	AIzaSy...
GOOGLE_SHEET_URL	Published CSV URL of Google Sheet	https://docs.google.com/...
WP_ENDPOINT	WordPress REST API endpoint	https://yoursite.com/wp-json/wp/v2/posts
WP_USERNAME	WordPress admin username	admin
WP_APPLICATION_PASSWORD	WordPress application password	xxxx xxxx xxxx xxxx xxxx
WP_CATEGORY_ID	Default category ID for posts	1
Google Sheets Format 📊

Create a Google Sheet with this structure:

Topic	URLs
Quantum Computing	https://example.com/qc\|https://...
AI Ethics	https://example.com/ethics
Web3 Security	

Publish as: File → Share → Publish to web → CSV

Usage 🚀

Run manually:

python publish_article.py


Scheduled execution (via GitHub Actions):

Daily at 7:30 AM EST (UTC 12:30)
Manual trigger through GitHub UI
GitHub Actions Setup ⚡

Add these secrets to your repository:

GEMINI_API_KEY
GOOGLE_SHEET_URL
WP_ENDPOINT
WP_USERNAME
WP_APP_PASSWORD
WP_CATEGORY_ID

The workflow will:

Install dependencies
Generate article using Gemini AI
Validate and format content
Publish to WordPress
Handle errors gracefully
Error Handling 🛡️

The system includes:

API rate limiting protection
CSV structure validation
URL sanitization
Content safety checks
Exponential backoff retries
Fallback topic system
Contributing 🤝
Fork the repository
Create feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request
