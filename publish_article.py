import os
import re
import csv
import time
import requests
import google.generativeai as genai
from datetime import datetime
from io import StringIO
from urllib.parse import urlparse
from typing import List, Dict, Optional
import warnings

# Suppress Gemini API warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.generativeai")

# Configuration constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
BACKOFF_FACTOR = 2
MAX_TOPIC_LENGTH = 200
MAX_URLS = 5

def get_article_topics(sheet_url: str) -> List[Dict]:
    """Fetch topics and URLs from Google Sheet with enhanced error handling"""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                sheet_url,
                headers={'User-Agent': 'AutoPublisher/1.0'},
                timeout=15
            )

            if response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_DELAY * (BACKOFF_FACTOR ** attempt)
                    print(f"Rate limited. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                raise requests.exceptions.HTTPError("Google Sheets API rate limit exceeded")

            response.raise_for_status()

            try:
                csv_content = StringIO(response.text)
                reader = csv.DictReader(csv_content)

                if not reader.fieldnames:
                    raise ValueError("Empty CSV file")

                if 'Topic' not in reader.fieldnames:
                    raise ValueError("CSV missing required 'Topic' column")

            except csv.Error as e:
                raise ValueError(f"Invalid CSV structure: {str(e)}") from e

            topics = []
            for row_num, row in enumerate(reader, start=2):
                try:
                    topic = row.get('Topic', '').strip()
                    if not topic:
                        print(f"Skipping row {row_num}: Empty topic")
                        continue

                    if len(topic) > MAX_TOPIC_LENGTH:
                        print(f"Skipping row {row_num}: Topic exceeds {MAX_TOPIC_LENGTH} characters")
                        continue

                    url_list = []
                    raw_urls = row.get('URLs', '')
                    if raw_urls:
                        for url in raw_urls.split('|'):
                            url = url.strip()
                            if url:
                                try:
                                    parsed = urlparse(url)
                                    if all([parsed.scheme, parsed.netloc]):
                                        if len(url_list) < MAX_URLS:
                                            url_list.append(url)
                                        else:
                                            print(f"Row {row_num}: Exceeded max URLs ({MAX_URLS}), truncating")
                                    else:
                                        print(f"Row {row_num}: Invalid URL scheme - {url}")
                                except ValueError:
                                    print(f"Row {row_num}: Malformed URL - {url}")

                    topics.append({
                        'topic': topic,
                        'urls': url_list
                    })

                except Exception as e:
                    print(f"Error processing row {row_num}: {str(e)}")
                    continue

            if not topics:
                raise ValueError("No valid topics found in sheet")

            return topics

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return []
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return []

    return []

def validate_topic_data(topic_data: Dict) -> bool:
    """Validate topic data structure"""
    if not isinstance(topic_data, dict):
        print("Invalid topic data format")
        return False

    required_fields = {
        'topic': (str, MAX_TOPIC_LENGTH),
        'urls': (list, MAX_URLS)
    }

    for field, (data_type, max_length) in required_fields.items():
        if field not in topic_data:
            print(f"Missing required field: {field}")
            return False
        if not isinstance(topic_data[field], data_type):
            print(f"Invalid type for {field}: {type(topic_data[field])}")
            return False
        if data_type == str and not 0 < len(topic_data[field]) <= max_length:
            print(f"Invalid length for {field}: {len(topic_data[field])}")
            return False
        if data_type == list and len(topic_data[field]) > max_length:
            print(f"Too many items in {field}: {len(topic_data[field])}")
            return False

    return True

def generate_markdown_article(topic_data: Dict) -> Optional[str]:
    """Generate article with error handling and safety checks"""
    if not validate_topic_data(topic_data):
        raise ValueError("Invalid topic data structure")

    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-pro')

    topic = topic_data['topic']
    urls = topic_data['urls']

    try:
        links_instruction = ""
        if urls:
            links_list = '\n'.join([f"[{i+1}] {url}" for i, url in enumerate(urls)])
            links_instruction = f"""Include 2-4 contextual links in the article body using markdown format.
            Use these sources where appropriate:
            {links_list}
            Add a references section at the end with corresponding numbers."""
        else:
            print("No valid URLs provided - generating without external links")

        prompt = f"""Write a comprehensive 800-word technical article about {topic} using Markdown.
        Follow this structure:

        # [Specific, Technical Title About {topic}]

        ## Introduction
        [Technical overview with industry context]

        ## Technical Details
        [In-depth technical analysis with specifications]

        ## Implementation Challenges
        [Technical hurdles and solutions]

        ## Best Practices
        [Expert recommendations and methodologies]

        ## Future Developments
        [Emerging technologies and research directions]

        {links_instruction if urls else ''}

        ## SEO Keywords
        - Keyword1
        - Keyword2
        - Keyword3

        Formatting Requirements:
        - Use H2 headings for main sections
        - **Bold** key technical terms
        - Use bullet points for lists
        - Include code snippets where applicable
        - Maintain technical accuracy
        - Cite sources when using specific data"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=4000
            ),
            safety_settings={
                'HARASSMENT': 'block_none',
                'HATE_SPEECH': 'block_none',
                'SEXUAL': 'block_none',
                'DANGEROUS': 'block_none'
            }
        )

        if not response.text:
            raise ValueError("Empty response from API")

        return response.text

    except genai.types.BlockedPromptError:
        print("Content generation blocked due to safety concerns")
        return None
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None

def add_references_section(content: str, urls: List[str]) -> str:
    """Add formatted references section with validation"""
    if not urls:
        return content

    try:
        references = "\n\n## References\n" + '\n'.join(
            [f"{i+1}. [{urlparse(url).netloc}]({url})" for i, url in enumerate(urls)]
        )

        if '## References' in content:
            return content

        return content.replace('## SEO Keywords', f"{references}\n\n## SEO Keywords")
    except Exception as e:
        print(f"Failed to add references: {str(e)}")
        return content

def post_to_wordpress(content: str) -> bool:
    """Post article to WordPress with validation"""
    if not content:
        raise ValueError("Empty content provided")

    try:
        title = content.split('\n')[0].replace('#', '').strip()
        if not title:
            raise ValueError("Missing article title")

        # Convert and validate IDs
        try:
            category_id = int(os.getenv('WP_CATEGORY_ID', '1'))
            tag_id = int(os.getenv('WP_TAG_ID', '13'))
        except ValueError as e:
            raise ValueError(f"Invalid ID format: {str(e)}") from e

        wp_endpoint = os.environ['WP_ENDPOINT']
        auth = (
            os.environ['WP_USERNAME'],
            os.environ['WP_APPLICATION_PASSWORD']
        )

        payload = {
            "title": title,
            "content": f"<!--markdown-->\n{content}",
            "status": "publish",
            "categories": [category_id],
            "tags": [tag_id],  # Now using validated integer ID
            "date": datetime.now().isoformat()
        }

        print(f"Debug: Final WordPress payload: {payload}")

        response = requests.post(
            wp_endpoint,
            auth=auth,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return True

    except requests.exceptions.RequestException as e:
        print(f"WordPress API Error: {str(e)}")
        if e.response is not None:
            print(f"Response body: {e.response.text}")
        return False
    except Exception as e:
        print(f"Posting error: {str(e)}")
        return False

def extract_seo_keywords(content: str) -> List[str]:
    """Extract SEO keywords from content"""
    try:
        keywords_section = content.split("## SEO Keywords")[1]
        return [
            line.replace('-', '').strip()
            for line in keywords_section.split('\n')
            if line.startswith('-')
        ][:5]
    except Exception as e:
        print(f"Keyword extraction error: {str(e)}")
        return []

def main():
    """Main execution flow with error handling"""
    try:
        print("Initializing automated publisher...")

        sheet_url = os.environ.get('GOOGLE_SHEET_URL')
        if not sheet_url:
            raise ValueError("Missing Google Sheet URL in environment variables")

        print("Fetching topics from Google Sheet...")
        topics = get_article_topics(sheet_url)

        if not topics:
            print("No valid topics found, using fallback")
            topics = [{
                'topic': 'Recent Advances in Artificial Intelligence',
                'urls': ['https://example.com/ai-advances']
            }]

        selected_topic = None
        for topic in topics:
            if validate_topic_data(topic):
                selected_topic = topic
                break

        if not selected_topic:
            raise ValueError("No valid topics available for processing")

        print(f"Selected topic: {selected_topic['topic']}")
        if selected_topic['urls']:
            print(f"Including {len(selected_topic['urls'])} reference URLs")

        print("\nGenerating article...")
        article = generate_markdown_article(selected_topic)

        if not article:
            raise ValueError("Article generation failed")

        print("\nPost-processing content...")
        processed_article = add_references_section(article, selected_topic['urls'])

        print("\nValidating final content...")
        if "## SEO Keywords" not in processed_article:
            print("Warning: SEO keywords section missing")

        print("\nPosting to WordPress...")
        success = post_to_wordpress(processed_article)

        if success:
            print("\nArticle published successfully!")
        else:
            print("\nPublication failed")

    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
