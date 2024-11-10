import requests
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd
from urllib.parse import urljoin
import re
import xml.etree.ElementTree as ET
import signal
import sys
from contextlib import contextmanager
import os
from pathlib import Path
from typing import List, Optional, Union, Dict
from urllib.parse import urljoin, urlparse
import hashlib
from datetime import datetime
import mimetypes
import time


def search_pmc(query, start=0, batch_size=20):
    """
    Search PMC database using E-utilities with pagination

    Parameters:
    query (str): Search query
    start (int): Starting index for results
    batch_size (int): Number of results per batch

    Returns:
    tuple: (list of results, total count of results)
    """
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Search parameters
    search_params = {
        'db': 'pmc',
        'term': query,
        'retstart': start,
        'retmax': batch_size,
        'usehistory': 'y',
        'retmode': 'xml'
    }

    try:
        # Perform initial search to get IDs
        search_response = requests.get(esearch_url, params=search_params)
        search_response.raise_for_status()

        # Parse search results
        search_tree = ET.fromstring(search_response.content)
        total_results = int(search_tree.find('.//Count').text)

        if total_results == 0:
            return [], 0

        # Get WebEnv and QueryKey for batch fetch
        webenv = search_tree.find('.//WebEnv').text
        query_key = search_tree.find('.//QueryKey').text

        # Fetch article details
        fetch_params = {
            'db': 'pmc',
            'WebEnv': webenv,
            'query_key': query_key,
            'retstart': start,
            'retmax': batch_size,
            'retmode': 'xml'
        }

        fetch_response = requests.get(efetch_url, params=fetch_params)
        fetch_response.raise_for_status()

        # Parse article details
        articles = []
        root = ET.fromstring(fetch_response.content)

        for article in root.findall('.//article'):
            try:
                pmcid = article.find('.//article-id[@pub-id-type="pmc"]').text
                articles.append({
                    'pmcid': pmcid,
                    'pmc_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}"
                })
            except AttributeError:
                continue

        return articles, total_results

    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {e}")
        return [], 0
    except ET.ParseError as e:
        print(f"Error parsing XML response: {e}")
        return [], 0

def fetch_all_results(query, batch_size=20):
    """
    Fetch all results using pagination

    Parameters:
    query (str): Search query
    batch_size (int): Number of results per batch

    Yields:
    dict: Article information (one at a time)
    """
    start = 0
    total_results = None

    while True:
        # Add delay to respect NCBI's rate limits
        time.sleep(0.34)  # Maximum 3 requests per second

        articles, total = search_pmc(query, start, batch_size)

        if total_results is None:
            total_results = total
            print(f"Total results found: {total_results}")

        if not articles:
            break

        for article in articles:
            yield article

        start += batch_size
        if start >= total_results:
            break

        print(f"Fetched {start} of {total_results} results...")

class GracefulInterruptHandler:
    def __init__(self):
        self.interrupted = False
        self.released = False
        # Store the original handlers
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)

    def __enter__(self):
        # Set up the new handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        return self

    def __exit__(self, type, value, traceback):
        # Restore the original handlers
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)
        self.released = True

    def signal_handler(self, signum, frame):
        self.interrupted = True
        print("\nInterrupt received! Finishing current figure and saving data...")
        if self.released:
            # If this handler is no longer supposed to be in effect,
            # call the original handler
            if signum == signal.SIGINT:
                self.original_sigint(signum, frame)
            elif signum == signal.SIGTERM:
                self.original_sigterm(signum, frame)

class ImageDownloader:
    def __init__(self, output_dir: str = "downloaded_images",
                 create_subdirs: bool = True,
                 delay: float = 1.0):
        """
        Initialize the image downloader.

        Args:
            output_dir (str): Base directory for downloaded images
            create_subdirs (bool): Create date-based subdirectories
            delay (float): Delay between downloads in seconds
        """
        self.base_dir = Path(output_dir)
        self.create_subdirs = create_subdirs
        self.delay = delay
        self.session = requests.Session()
        # Set a user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _create_output_dir(self) -> Path:
        """Create output directory with optional date-based subdirectory."""
        if self.create_subdirs:
            date_dir = datetime.now().strftime('%Y-%m-%d')
            output_dir = self.base_dir / date_dir
        else:
            output_dir = self.base_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_file_extension(self, image_name: str, url: str, content_type: Optional[str] = None) -> str:
        """Get file extension from URL or content-type."""
        return f"{image_name}.jpg"

    def _generate_filename(self, image_name: str, url: str, content_type: Optional[str] = None) -> str:
        """Generate a unique filename based on URL and timestamp."""
        return self._get_file_extension(image_name, url, content_type)

    def find_images(self, soup: BeautifulSoup, classes: List[str],
                    base_url: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Find images in soup with specified classes.

        Args:
            soup: BeautifulSoup object
            classes: List of class names to search for
            base_url: Base URL for relative image URLs

        Returns:
            List of dicts containing image info
        """
        images = []
        for class_name in classes:
            found = soup.find_all('img', class_=class_name)
            for img in found:
                src = img.get('src')
                if src:
                    if base_url and not src.startswith(('http://', 'https://')):
                        src = urljoin(base_url, src)
                    images.append({
                        'url': src,
                        'alt': img.get('alt', ''),
                        'class': class_name,
                        'original_src': img.get('src')
                    })
        return images

    def download_image(self, image_name: str, url: str, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Download an image and save it to the output directory.

        Args:
            url: Image URL to download
            output_dir: Optional specific output directory

        Returns:
            Dict with download results
        """
        if output_dir is None:
            output_dir = self._create_output_dir()

        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('content-type')
            filename = self._generate_filename(image_name, url, content_type)
            filepath = output_dir / filename

            # Download the image in chunks
            print(f"Downloading {url}: into {filepath}")
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return {
                'status': 'success',
                'url': url,
                'filepath': str(filepath),
                'filename': filename,
                'content_type': content_type
            }

        except Exception as e:
            return {
                'status': 'error',
                'url': url,
                'error': str(e)
            }

    def download_images(self, image_name, soup: BeautifulSoup, classes: List[str],
                       base_url: Optional[str] = None) -> Dict[str, str]:
        """
        Find and download all matching images.
        Args:

            soup: BeautifulSoup object
            classes: List of class names to search for
            base_url: Base URL for relative image URLs

        Returns:
            List of download results
        """
        output_dir = self._create_output_dir()
        images = self.find_images(soup, classes, base_url)

        if len(images) < 1:
            return []
        img = images[0]

        result = self.download_image(image_name, img['url'], output_dir)
        return result['filename']

class PMCScraper:
    def __init__(self, email, tool_name="PMCFigureScraper"):
        self.headers = {
            'User-Agent': f'{tool_name}/1.0 (https://example.com; mailto:{email})',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_figure_urls(self, article_url):
        try:
            response = self.session.get(article_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            article_url = article_url if article_url.endswith("/") else f"{article_url}/"
            figure_links = {}
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'figure/' in href:
                    full_url = article_url + href
                    key = href[len('figure/'):]
                    if key.endswith('/'):
                        key = key[:-1]
                    figure_links[key] = full_url

            return figure_links

        except requests.exceptions.RequestException as e:
            print(f"Error fetching article page {article_url}: {e}")
            print(f"Response status code: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"Response headers: {getattr(e.response, 'headers', {})}")
            return []

    def get_figure_caption(self, figure_url):
        try:
            response = self.session.get(figure_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            caption_div = (
                soup.find('figcaption') or
                soup.find('div', class_=['caption', 'fig-caption']) or
                soup.find('div', {'id': 'caption'}) or
                soup.find('div', class_='fig-caption-inner')
            )

            if caption_div:
                caption_text = caption_div.get_text(strip=True)
                caption_text = re.sub(r'\s+', ' ', caption_text)
                return caption_text, soup

            return None, None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching figure page {figure_url}: {e}")
            return None, None

    def get_figure_image(self, image_name, soup, output_dir):
        downloader = ImageDownloader(output_dir, create_subdirs=True)
        try:
            result = downloader.download_images(
                image_name=image_name,
                soup=soup,
                classes=['graphic', 'zoom-in'],
                base_url='https://cdn.ncbi.nlm.nih.gov'
            )

            return result

        except requests.exceptions.RequestException as e:
            print(f"Error fetching figure page {figure_url}: {e}")
            return None

def save_progress(processed_pmcids, output_file):
    """Save the list of processed PMCIDs to a file"""
    with open(f"{output_file}.progress", 'w') as f:
        f.write('\n'.join(processed_pmcids))

def load_progress(output_file):
    """Load the list of processed PMCIDs from a file"""
    try:
        with open(f"{output_file}.progress", 'r') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        return set()

def process_articles_and_save_captions(query, email, output_file='figure_captions.csv', output_dir='image'):
    scraper = PMCScraper(email=email)
    headers = ['pmcid', 'figureid', 'image', 'caption']

    # Load progress from previous run
    processed_pmcids = load_progress(output_file)
    print(f"Found {len(processed_pmcids)} previously processed articles")

    # Open CSV file in append mode if there's previous progress
    mode = 'a' if processed_pmcids else 'w'

    with GracefulInterruptHandler() as handler:
        with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if mode == 'w':
                writer.writeheader()

            try:
                for article in fetch_all_results(query):
                    pmcid = article['pmcid']

                    # Skip if already processed
                    if pmcid in processed_pmcids:
                        print(f"Skipping already processed article {pmcid}")
                        continue

                    article_url = article['pmc_url']
                    print(f"\nProcessing article {pmcid}...")

                    figure_urls = scraper.get_figure_urls(article_url)

                    if figure_urls:
                        print(f"Found {len(figure_urls)} figure links")

                        for (figureid, figure_url)  in figure_urls.items():
                            if handler.interrupted:
                                print("Interrupt confirmed - saving progress and exiting...")
                                break

                            print(f"Processing figure: {figure_url}")

                            caption, soup = scraper.get_figure_caption(figure_url)
                            if caption is None or len(caption) < 50:
                                time.sleep(0.34)
                                continue

                            if 'CT' not in caption or \
                                ('Spine' not in caption and 'spine' not in caption):
                                time.sleep(0.34)
                                continue

                            image = scraper.get_figure_image(f"{pmcid}_{figureid}", soup, output_dir)

                            if caption:
                                writer.writerow({
                                    'pmcid': pmcid,
                                    'figureid': figureid,
                                    'image': image,
                                    'caption': caption
                                })
                                csvfile.flush()  # Ensure immediate write

                            time.sleep(0.34)
                    else:
                        print("No figure links found")

                    # Mark article as processed and save progress
                    processed_pmcids.add(pmcid)
                    save_progress(processed_pmcids, output_file)

                    if handler.interrupted:
                        break

            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
            finally:
                # Save final progress
                save_progress(processed_pmcids, output_file)
                print("\nProgress saved. You can resume from where you left off by running the script again.")

def main():
    query = "Spine CT[Figure/Table Caption]"
    output_file = 'cervical_spine_figures-3.csv'
    output_dir = 'image_data_3'
    email = "your.email@example.com"  # Replace with your email

    print(f"Searching PMC for: {query}")
    print(f"Results will be saved to: {output_file}")

    try:
        process_articles_and_save_captions(query, email, output_file, output_dir)

        # Display summary of results
        df = pd.read_csv(output_file)
        print("\nFinal Summary:")
        print(f"Total articles processed: {df['pmcid'].nunique()}")
        print(f"Total figures found: {len(df)}")
        print(f"Average figures per article: {len(df) / df['pmcid'].nunique():.1f}")
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

