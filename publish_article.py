import os
import csv
import time
import requests
import google.generativeai as genai
from datetime import datetime
from io import StringIO
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Optional
import warnings
import git  # Import GitPython
import re  # Import the regular expression module

# Suppress Gemini API warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.generativeai")

# Configuration constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
BACKOFF_FACTOR = 2
MAX_TOPIC_LENGTH = 200
MAX_URLS = 5  # Maximum URLs to *process*, even if more are present
DEFAULT_CATEGORY_ID = 1  # You can remove this if you don't use it
DEFAULT_TAG_ID = 13  # You can remove this if you don't use it

# --- CRITICAL: CHANGE THESE ---
GIT_REPO_PATH = "/path/to/your/local/repo"  #  **ABSOLUTE PATH** to your local Git repo
GIT_POSTS_FOLDER = "docs"  # Folder where Markdown files should go (check your plugin settings)
IMAGE_FOLDER = "_images" # As per the plugin documentation

def get_article_topics(sheet_url: str) -> List[Dict]:
    """Fetches topics and URLs from a Google Sheet (published as CSV)."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount('https://', adapter)

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                sheet_url,
                headers={'User-Agent': 'AutoPublisher/1.0'},  # Good practice
                timeout=15  # Reasonable timeout
            )

            if response.status_code == 429:  # Rate limiting
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_DELAY * (BACKOFF_FACTOR ** attempt)
                    print(f"Rate limited. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                raise requests.exceptions.HTTPError("Google Sheets API rate limit exceeded")

            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            csv_content = StringIO(response.text)
            reader = csv.DictReader(csv_content)

            if not reader.fieldnames:
                raise ValueError("Empty CSV file")

            if 'Topic' not in reader.fieldnames:
                raise ValueError("CSV missing required 'Topic' column")

            topics = []
            for row_num, row in enumerate(reader, start=2):  # Start at 2 for human-readable row numbers
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
                        if not url:  # Skip empty URLs
                            continue
                        try:
                            parsed = urlparse(url)
                            if parsed.scheme and parsed.netloc:  # Check for valid scheme and netloc
                                if len(url_list) < MAX_URLS:
                                    url_list.append(url)
                                else:
                                    print(f"Row {row_num}: Exceeded max URLs ({MAX_URLS}), truncating")
                                    break  # Stop processing URLs for this row
                            else:
                                print(f"Row {row_num}: Invalid URL scheme - {url}")
                        except ValueError:
                            print(f"Row {row_num}: Malformed URL - {url}")

                topics.append({
                    'topic': topic,
                    'urls': url_list
                })

            if not topics:
                raise ValueError("No valid topics found in sheet")

            return topics

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)  # Wait before retrying
            else:
                return []  # Return empty list after all retries fail

        except (ValueError, csv.Error) as e:
            print(f"Processing error: {e}")  # Catch CSV and general value errors
            return []

    return []  # Return empty list if all attempts fail


def validate_topic_data(topic_data: Dict) -> bool:
    """Validates the structure and content of a topic data dictionary."""
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


def generate_markdown_article(topic_data: Dict) -> Optional[tuple[str, str]]:
    """Generates a Markdown article using Google Gemini and returns the content and title."""
    if not validate_topic_data(topic_data):
        raise ValueError("Invalid topic data structure")

    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-pro')

    topic = topic_data['topic']
    urls = topic_data['urls']

    try:
        if urls:
            links_list = '\n'.join([f"[{i+1}] {url}" for i, url in enumerate(urls)])
            links_instruction = f"""Include 2-4 contextual links in the article body using markdown format.
            Use these sources where appropriate:
            {links_list}
            Add a references section at the end with corresponding numbers."""
        else:
            links_instruction = ""
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
                temperature=0.7,  # Adjust as needed
                top_p=0.9,        # Adjust as needed
                max_output_tokens=4000  # Increased for longer articles
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

        # Extract the title from the generated Markdown
        match = re.search(r"^#\s+(.+)$", response.text, re.MULTILINE)
        if match:
            title = match.group(1).strip()
        else:
            title = None  # Use the topic as a fallback if no title is found
            print("Warning: Could not extract title from Markdown. Using topic as filename.")

        return response.text, title


    except genai.types.BlockedPromptError as e:
        print(f"Content generation blocked: {e}")
        return None, None  # Return None for both content and title
    except Exception as e:
        print(f"Generation error: {e}")
        return None, None  # Return None for both content and title


def add_references_section(content: str, urls: List[str]) -> str:
    """Adds a formatted references section to the Markdown content."""
    if not urls:
        return content  # Return original content if no URLs

    references = "\n\n## References\n" + '\n'.join(
        [f"{i+1}. [{urlparse(url).netloc}]({url})" for i, url in enumerate(urls)]
    )
    # Replace only the FIRST occurrence of "## SEO Keywords"
    return content.replace('## SEO Keywords', f"{references}\n\n## SEO Keywords", 1)


def extract_seo_keywords(content: str) -> List[str]:
    """Extracts SEO keywords from the Markdown content."""
    try:
        keywords_section = content.split("## SEO Keywords")[1]
        keywords = [line.replace('-', '').strip() for line in keywords_section.split('\n') if line.startswith('-')]
        return keywords[:5]  # Limit to 5 keywords
    except IndexError:
        print("SEO Keywords section not found or malformed.")
        return []
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []


def save_markdown_to_git(markdown_content: str, title: Optional[str], topic: str) -> bool:
    """Saves the Markdown content to a file in the Git repository."""
    try:
        # Use the title for the filename if available, otherwise fall back to the topic
        if title:
            file_slug = quote_plus(title.lower().replace(" ", "-"))
        else:
            file_slug = quote_plus(topic.lower().replace(" ", "-"))

        filename = f"{file_slug}.md"
        filepath = os.path.join(GIT_REPO_PATH, GIT_POSTS_FOLDER, filename)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            f.write(markdown_content)
        print(f"Markdown saved to: {filepath}")
        return True

    except Exception as e:
        print(f"Error saving Markdown to file: {e}")
        return False


def commit_and_push(message: str) -> bool:
    """Commits and pushes changes to the Git repository."""
    try:
        repo = git.Repo(GIT_REPO_PATH)
        repo.git.add(A=True)  # Add all changes (including new files)
        repo.git.commit(m=message)
        origin = repo.remote(name='origin')
        origin.push()
        print("Changes pushed to GitHub.")
        return True
    except git.exc.GitCommandError as e:
        print(f"Git command error: {e}")
        return False
    except Exception as e:
        print(f"Error during commit/push: {e}")
        return False


def main():
    """Main function to orchestrate article generation and publishing."""
    try:
        print("Initializing automated publisher...")

        sheet_url = os.environ.get('GOOGLE_SHEET_URL')
        if not sheet_url:
            raise ValueError("Missing Google Sheet URL in environment variables")

        print("Fetching topics from Google Sheet...")
        topics = get_article_topics(sheet_url)

        # Fallback topic if no topics are found or all are invalid
        if not topics:
            print("No valid topics found, using fallback")
            topics = [{
                'topic': 'Recent Advances in Artificial Intelligence',  # A reasonable fallback
                'urls': []  # No URLs for the fallback
            }]

        # Select the *first* valid topic
        selected_topic = None
        for topic in topics:
            if validate_topic_data(topic):
                selected_topic = topic
                break  # Stop after finding the first valid topic

        if not selected_topic:
            raise ValueError("No valid topics available for processing")

        print(f"Selected topic: {selected_topic['topic']}")
        if selected_topic['urls']:
            print(f"Including {len(selected_topic['urls'])} reference URLs")
        else:
            print("No reference URLs provided.")

        print("\nGenerating article...")
        article, title = generate_markdown_article(selected_topic)  # Get both content and title

        if not article:
            raise ValueError("Article generation failed")

        print("\nPost-processing content...")
        processed_article = add_references_section(article, selected_topic['urls'])

        print("\nValidating final content...")
        # Basic validation - check for the presence of the SEO Keywords section
        if "## SEO Keywords" not in processed_article:
            print("Warning: SEO keywords section missing")  # Warn, but don't necessarily fail

        print("\nSaving Markdown to Git repository...")
        # Pass the title to save_markdown_to_git
        if not save_markdown_to_git(processed_article, title, selected_topic['topic']):
            raise ValueError("Failed to save Markdown to Git")


        print("\nCommitting and pushing changes...")
        # Use the title in the commit message if available, otherwise the topic
        commit_message = f"Add article: {title if title else selected_topic['topic']}"
        if not commit_and_push(commit_message):
            raise ValueError("Git commit and push failed")

        print("Article will be published automatically via webhook.")

    except Exception as e:
        print(f"\nCritical error: {e}")
        raise  # Re-raise the exception to stop execution


if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# file_name.md
