import os
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
MAX_URLS = 5  # Maximum URLs to *process*, even if more are present
DEFAULT_CATEGORY_ID = 1
DEFAULT_TAG_ID = 13


def get_article_topics(sheet_url: str) -> List[Dict]:
    """
    Fetches topics and URLs from a Google Sheet.  URLs are now optional.

    Args:
        sheet_url: The URL of the Google Sheet.

    Returns:
        A list of dictionaries, where each dictionary represents a topic
        and its associated URLs (which may be an empty list).
        Returns an empty list on failure.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
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

            csv_content = StringIO(response.text)
            reader = csv.DictReader(csv_content)

            if not reader.fieldnames:
                raise ValueError("Empty CSV file")

            if 'Topic' not in reader.fieldnames:
                raise ValueError("CSV missing required 'Topic' column")

            topics = []
            for row_num, row in enumerate(reader, start=2):
                topic = row.get('Topic', '').strip()
                if not topic:
                    print(f"Skipping row {row_num}: Empty topic")
                    continue

                if len(topic) > MAX_TOPIC_LENGTH:
                    print(f"Skipping row {row_num}: Topic exceeds {MAX_TOPIC_LENGTH} characters")
                    continue

                url_list = []
                raw_urls = row.get('URLs', '')
                if raw_urls:  # Only process URLs if the field is not empty
                    for url in raw_urls.split('|'):
                        url = url.strip()
                        if not url:  # Skip empty URLs after splitting
                            continue
                        try:
                            parsed = urlparse(url)
                            if parsed.scheme and parsed.netloc:
                                if len(url_list) < MAX_URLS:
                                    url_list.append(url)
                                else:
                                    print(f"Row {row_num}: Exceeded max URLs ({MAX_URLS}), truncating")
                                    break  # Stop processing URLs for this row
                            else:
                                print(f"Row {row_num}: Invalid URL scheme - {url}")
                        except ValueError:
                            print(f"Row {row_num}: Malformed URL - {url}")
                # else:  # No 'else' needed - url_list will be empty by default

                topics.append({
                    'topic': topic,
                    'urls': url_list  # Always add the URL list, even if empty
                })

            if not topics:
                raise ValueError("No valid topics found in sheet")

            return topics

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return []

        except (ValueError, csv.Error) as e:
            print(f"Processing error: {e}")
            return []

    return []


def validate_topic_data(topic_data: Dict) -> bool:
    """
    Validates the structure and content of a topic data dictionary.
    URLs are now considered optional (can be an empty list).

    Args:
        topic_data: A dictionary containing topic information.

    Returns:
        True if the data is valid, False otherwise.
    """
    if not isinstance(topic_data, dict):
        print("Invalid topic data format")
        return False

    if 'topic' not in topic_data or not isinstance(topic_data['topic'], str) or not topic_data['topic'].strip():
        print("Missing or invalid 'topic' field.")
        return False

    if len(topic_data['topic']) > MAX_TOPIC_LENGTH:
        print(f"Topic exceeds maximum length ({MAX_TOPIC_LENGTH} characters).")
        return False

    if 'urls' not in topic_data or not isinstance(topic_data['urls'], list):
        print("Missing or invalid 'urls' field.")
        return False

    # No length check for 'urls' - it can be empty
    # We still validate the *contents* of the list if it's not empty

    for url in topic_data['urls']:
        if not isinstance(url, str):
            print(f"Invalid URL type: {type(url)}")
            return False
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                print(f"Invalid URL: {url}")
                return False
        except ValueError:
            print(f"Malformed URL: {url}")
            return False

    return True


def generate_markdown_article(topic_data: Dict) -> Optional[str]:
    """
    Generates a Markdown article using the Gemini API.  Handles optional URLs.

    Args:
        topic_data: A dictionary containing the topic and URLs (which may be empty).

    Returns:
        The generated Markdown article as a string, or None on failure.
    """
    if not validate_topic_data(topic_data):
        raise ValueError("Invalid topic data structure")

    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-pro')

    topic = topic_data['topic']
    urls = topic_data['urls']  # URLs can now be an empty list

    try:
        if urls:  # Only create the links instruction if there are URLs
            links_list = '\n'.join([f"[{i+1}] {url}" for i, url in enumerate(urls)])
            links_instruction = f"""Include 2-4 contextual links in the article body using markdown format.
            Use these sources where appropriate:
            {links_list}
            Add a references section at the end with corresponding numbers."""
        else:
            links_instruction = ""  # No links instruction if no URLs
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

        {links_instruction}

        ## SEO Keywords  <-- MUST BE INCLUDED
        - Keyword1
        - Keyword2
        - Keyword3
        - Keyword4
        - Keyword5

        You MUST include an "SEO Keywords" section with exactly 5 keywords, each on a new line, starting with a hyphen and a space.  Example:
        - DengueFever
        - MosquitoControl
        - MarylandHealth
        - VectorBorne
        - PublicHealth

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

    except genai.types.BlockedPromptError as e:
        print(f"Content generation blocked: {e}")
        return None
    except Exception as e:
        print(f"Generation error: {e}")
        return None


def add_references_section(content: str, urls: List[str]) -> str:
    """
    Adds a formatted references section to the Markdown content (if URLs exist).

    Args:
        content: The Markdown content.
        urls: A list of URLs to include in the references (can be empty).

    Returns:
        The content with the added references section (or original if no URLs).
    """
    if not urls:  # No references if the URL list is empty
        return content

    references = "\n\n## References\n" + '\n'.join(
        [f"{i+1}. [{urlparse(url).netloc}]({url})" for i, url in enumerate(urls)]
    )
    return content.replace('## SEO Keywords', f"{references}\n\n## SEO Keywords")


def extract_seo_keywords(content: str) -> List[str]:
    """Extracts SEO keywords from the generated content."""
    try:
        keywords_section = content.split("## SEO Keywords")[1]
        keywords = [line.replace('-', '').strip() for line in keywords_section.split('\n') if line.startswith('-')]
        return keywords[:5]
    except IndexError:
        print("SEO Keywords section not found or malformed.")
        return []
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []


def main():
    """Main function to orchestrate the article generation and posting."""
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
                'urls': []  # Empty URL list for the fallback
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
        else:
            print("No reference URLs provided.")  # Indicate when no URLs are used

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
        print(f"\nCritical error: {e}")
        raise


if __name__ == "__main__":
    main()
