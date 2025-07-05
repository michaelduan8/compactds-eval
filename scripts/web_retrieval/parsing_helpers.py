import aiohttp
import hashlib
import os
import re
import string

from typing import Tuple

from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig,
    BrowserConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
    CrawlerMonitor,
)
from nltk.tokenize import sent_tokenize
from resiliparse.extract.html2text import extract_plain_text

import http.cookies
http.cookies._is_legal_key = lambda _: True

JINA_PREFIX = "https://r.jina.ai/"
JINA_API_KEY = os.getenv('JINA_API_KEY')

# TODO: consider merging fetch_and_parse_html with fetch_and_parse_jina
async def fetch_and_parse_html(session, url, progress_bar):
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.get(url, timeout=timeout) as response: 
            if response.ok:
                html = await response.text()

                progress_bar.update(1)

                try:
                    # Use resiliparse following DCLM ablations to extract web content
                    text = extract_plain_text(html, main_content=True)
                    if not text:
                        raise Exception("Resiliparse yielded empty text, trying with Beautiful Soup")
                    
                    return url, text, "Success!", "resiliparse" 
                
                except Exception:
                    try:
                        parse_mode = 'lxml'
                        soup = BeautifulSoup(html, 'lxml')
                    except Exception:
                        parse_mode = 'html'
                        soup = BeautifulSoup(html, 'html.parser')
                    
                    return url, soup.get_text(separator=' ', strip=True), "Success!", parse_mode
            
            else:
                progress_bar.update(1)
                return url, None, f"Status Code error when requesting webpage: {response.status}", None
            
    except Exception as e:
        progress_bar.update(1)
        return url, None, f"Error requesting/parsing webpage: {f'{type(e)},{e}'}", None


async def fetch_and_parse_html_jina(session, url, progress_bar):
    try:
        timeout = aiohttp.ClientTimeout(total=120)
        
        # Prepend jina to url
        jina_headers = {
            'Authorization': f'Bearer {JINA_API_KEY}',
            'X-Return-Format': 'markdown',
            'X-Engine': 'browser'
        }
        jina_url = f"{JINA_PREFIX}{url}"
        async with session.get(jina_url, timeout=timeout, headers=jina_headers) as response: 
            # Check if the request was successful
            if response.ok:
                response = await response.text()

                progress_bar.update(1)

                return url, response, "Success!", "jina" 
            
            else:
                progress_bar.update(1)
                return url, None, f"Status Code error when requesting webpage: {response.status}", None
            
    except Exception as e:
        progress_bar.update(1)
        return url, None, f"Error requesting/parsing webpage: {f'{type(e)},{e}'}", None


async def fetch_and_parse_html_crawl4ai(urls, max_concurrency):
    config = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        semaphore_count=max_concurrency,      # Max concurrent requests
        stream=False
        # magic=True
    )

    browser_config = BrowserConfig(headless=True, verbose=False)

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrency,
        monitor=CrawlerMonitor()
    )

    out = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Get all results at once
        results = await crawler.arun_many(
            urls=urls,
            config=config,
            dispatcher=dispatcher
        )

        # Process all results after completion
        assert len(urls) == len(results)
        for url, result in zip(urls, results):
            if result.success:
                out.append((url, result.markdown, "Success!", "crawl4ai"))
            else:
                out.append((url, None, f"Failed to crawl {result.url}: {result.error_message}", None))
        
    return out


def _generate_url_hash(url, hash_algorithm='sha256'):
    # Choose the hash algorithm (e.g., sha256, md5, etc.)
    hash_func = getattr(hashlib, hash_algorithm)()
    
    # Encode the URL and generate the hash
    hash_func.update(url.encode('utf-8'))
    
    # Return the hash string
    return hash_func.hexdigest()


async def download_pdf(session, url, local_dir, progress_bar):
    # TODO: This needs to return output path of file
    try:
        # Send an asynchronous GET request to the URL
        timeout = aiohttp.ClientTimeout(total=300)
        async with session.get(url, timeout=timeout) as response:
            # Check if the request was successful
            if response.status == 200:
                filename = url.split("/")[-1]
                output_path = os.path.join(local_dir, f"{_generate_url_hash(url)}_{filename}")
                # Write the PDF content to a file
                with open(output_path, "wb") as f:
                    f.write(await response.read())
                progress_bar.update(1)  # Update the progress bar
                
                return url, output_path, "Success!"
            
            else:
                progress_bar.update(1)
                return url, None, f"Status Code error when downloading pdf: {response.status}"
            
    except Exception as e:
        progress_bar.update(1)
        return url, None, f"{url}, Error downloading pdf: {repr(e)}"


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def post_process_webpage(text, result, simple):
    # Logic borrowed from https://github.com/sunnynexus/Search-o1/blob/main/scripts/bing_search.py
    if simple:
        if "snippet" in result:
            snippet = result["snippet"]
            success, context = extract_snippet_with_context(text, snippet)
            if not success and "Failed to extract snippet context due to" in context:
                print("Critical failure during snippet extraction:")
                print(context)
                print(len(text.split()))
                quit()
            else:
                text = context
        else:
            text = text[:8000]

    # Remove any embedded urls
    pattern = r"\(https?:.*?\)|\[https?:.*?\]"
    text = re.sub(pattern, "", text).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')

    return text
