'''
Script to prepare data with retrieval results for evaluation, including reranking options
'''
import aiohttp
import argparse
import asyncio
import glob
import numpy as np
import os
import pdfplumber
import random
import time

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from modules.retriever.web_search.google_search import GoogleSearch
from src.utils import (
    load_json,
    load_jsonl,
    write
)
from scripts.web_retrieval.parsing_helpers import (
    fetch_and_parse_html,
    fetch_and_parse_html_jina,
    fetch_and_parse_html_crawl4ai,
    download_pdf,
    post_process_webpage
)

from urllib.parse import urlparse

import http.cookies
http.cookies._is_legal_key = lambda _: True


import logging
logger = logging.getLogger(__name__)


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    # 'Referer': 'https://www.google.com/',
    'Upgrade-Insecure-Requests': '1'
}


# TODO: for now, just checks google cse metadata
def is_pdf(result):
    link = result['link']
    pdf_mime = 'mime' in result and 'pdf' in result['mime'].lower()
    pdf_url = os.path.splitext(link)[1].lower() == '.pdf'

    return pdf_mime or pdf_url


async def download_pdfs(urls, local_dir, max_concurrency):
    # Create a tqdm progress bar
    url_to_success = {}
    url_to_out = {}
    with tqdm(total=len(urls), desc="Downloading PDFs", unit="file") as progress_bar:
        # ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(limit=max_concurrency) # ssl=ssl_context
        async with aiohttp.ClientSession(headers=HEADERS, timeout=None, connector=connector) as session:
            # Limit max_concurrency concurrent requests to match tcp connections
            sem = asyncio.Semaphore(max_concurrency)
            async def _download_pdf(session, url, local_dir, progress_bar):
                async with sem:
                    result = await download_pdf(session, url, local_dir, progress_bar)
                    time.sleep(0.2)  # Simple rate_limiting courtesy of search-o1
                    return result
                
            tasks = []
            for url in urls:
                task = asyncio.create_task(_download_pdf(session, url, local_dir, progress_bar))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            for url, output_path, message in results:
                url_to_success[url] = message
                url_to_out[url] = output_path

    error_counts = Counter([message for _, message in url_to_success.items() if "Success!" not in message])
    print(error_counts)
    print(f"Num response code error: {np.sum([1 for _, message in url_to_success.items() if 'Status Code error when downloading pdf:' in message])}")
    print(f"Num download error: {np.sum([1 for _, message in url_to_success.items() if 'Error downloading pdf:' in message])}")

    return url_to_out
    

async def get_raw_texts_from_html(urls, mode, max_concurrency):
    url_to_raw = {}
    url_to_success = {}
    parse_modes = []
    results = None
    if mode == 'crawl4ai':
        results = await fetch_and_parse_html_crawl4ai(urls, max_concurrency)
    else: 
        with tqdm(total=len(urls), desc="Retrieving raw content from webtext URLs...", unit="URL(s)") as progress_bar:
            # ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(limit=max_concurrency)
            async with aiohttp.ClientSession(headers=HEADERS, timeout=None, connector=connector) as session:
                # Limit max_concurrency concurrent requests to match tcp connections
                sem = asyncio.BoundedSemaphore(max_concurrency)
                async def _fetch_and_parse_html(session, url, progress_bar, mode):
                    async with sem:
                        if mode == 'jina':
                            result = await fetch_and_parse_html_jina(session, url, progress_bar)
                            # Wait longer with JINA to avoid API key rate limiting
                            time.sleep(1)
                        else:
                            result = await fetch_and_parse_html(session, url, progress_bar)
                            time.sleep(0.1)
                        return result
                    
                tasks = []
                for url in urls:
                    task = asyncio.create_task(_fetch_and_parse_html(session, url, progress_bar, mode))
                    tasks.append(task)
                results = await asyncio.gather(*tasks)

    assert results
    for url, content, message, parse_mode in results:
        if "Success!" in message:
            url_to_raw[url] = content
        url_to_success[url] = message
        parse_modes.append(parse_mode)


    # TODO: Map urls to output paths
    # error_counts = Counter([message for _, message in url_to_success.items() if "Error requesting/parsing webpage" in message]) # : <class 'RuntimeError'>
    # print(error_counts)
    # error_counts2 = Counter([message for _, message in url_to_success.items() if "Status Code error when requesting webpage:" in message]) # : <class 'RuntimeError'>
    # print(error_counts2)
    # bad_codes = [url for url, message in url_to_success.items() if "Status Code error when requesting webpage: 403" in message]
    # print(bad_codes[:10])
    # bad_codes_2 = [url for url, message in url_to_success.items() if "Error requesting/parsing webpage" in message]
    # print(bad_codes_2[:10])
    parse_counts = Counter(parse_modes)
    print(parse_counts)
    print(f"Num error fetching/parsing webpage: {np.sum([1 for _, content in url_to_success.items() if 'Error requesting/parsing webpage' in content])}")
    print(f"Num status code error: {np.sum([1 for _, content in url_to_success.items() if 'Status Code error when requesting webpage:' in content])}")

    return url_to_raw


def extract_pdf_metadata(parsed_pdf_dir, max_workers):
    pdf_metadatas = []
    parsed_pdf_metadatas = glob.glob(os.path.join(parsed_pdf_dir, "*"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_jsonl, file): file for file in parsed_pdf_metadatas}

        # Wrap the as_completed iterator with tqdm for progress tracking
        for future in tqdm(as_completed(future_to_file), total=len(parsed_pdf_metadatas), desc="Processing JSON files"):
            pdf_metadatas.extend(future.result())
    

    print(f"Number of parsed pdfs: {len(pdf_metadatas)}")
    return pdf_metadatas


def simple_pdf_extract(pdf_dir, max_workers):
    def _extract_pdf_text(pdf_path):
        try:
            # Open the PDF file using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
                    
                    if len(full_text.split()) > 600:
                        break
            
            # Limit the text length
            cleaned_text = ' '.join(full_text.split()[:600])
            # Hacky case logic for cases with extreme undertokenization when splitting on white space
            # Truncate to 10k characters
            cleaned_text = cleaned_text[:10000]

            return cleaned_text, "Success!"
        except Exception as e:
            return None, f"Error: {str(e)}"

    out_to_raw = {}
    pdf_paths = glob.glob(os.path.join(pdf_dir, '**', '*'), recursive=True)
    failure_count = 0
    # TODO: make max_workers a param
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(_extract_pdf_text, pdf_path): pdf_path
            for pdf_path in pdf_paths
        }
        for future in tqdm(as_completed(futures), desc="Extracting pdf text with pdfplumber", total=len(pdf_paths)):
            pdf_path = futures[future]
            simple_pdf_path = os.path.basename(urlparse(pdf_path).path)
            text, message = future.result()
            if message == "Success!":
                out_to_raw[simple_pdf_path] = text
            else:
                failure_count += 1
    
    print(f"Num pdfplumber extraction failures: {failure_count}")
    return out_to_raw, failure_count

    
def main(args): 
    random.seed(1)

    retrieval_queries = args.retrieval_queries

    retrieval_query_key = args.retrieval_query_key
    k = args.k
    num_tries = args.num_tries

    session_name = args.session_name
    intermediate_output_dir = args.intermediate_output_dir
    output_dir = args.output_dir

    links_only = args.links_only
    load_pdfs = args.load_pdfs
    load_webpages = args.load_webpages

    web_parse_mode = args.web_parse_mode

    simple = args.simple

    n_samples = args.n_samples
    full_session_name = f"{session_name}_top{k}{f'_{n_samples}' if n_samples else ''}"

    max_concurrency = args.max_concurrency
    max_workers = args.max_workers

    # Load retrieval queries
    queries = load_jsonl(retrieval_queries)
    if n_samples:
        random.shuffle(queries)
        queries = queries[:n_samples]

    print("Example query")
    print(queries[0])

    intermediate_results_path = os.path.join(intermediate_output_dir, full_session_name, "intermediate_result_links.jsonl")
    if os.path.exists(intermediate_results_path):
        query_results = load_jsonl(intermediate_results_path)
        remaining_queries = []
        for i, (query, result) in enumerate(zip(queries, query_results)):
            if len(result) == 0:
                remaining_queries.append((i, query))
    else:
        query_results = [[] for _ in range(len(queries))]
        remaining_queries = [(i, query) for i, query in enumerate(queries)]

    # Build Web Retrieval
    # TODO: currently only supports Google CSE
    web_search = GoogleSearch(retrieval_query_key, k)

    retries = 0
    while len(remaining_queries) > 0 and retries < num_tries:
        # Perform web search
        rqs = [q[1] for q in remaining_queries]
        results = web_search.retrieve(rqs)
        
        new_remaining_queries = []
        for (i, q), r in zip(remaining_queries, results):
            if len(r) == 0:
                new_remaining_queries.append((i, q))
            else:
                query_results[i] = r
        
        remaining_queries = new_remaining_queries
        retries += 1
    
    print(f"Number of queries with 0 results: {len(remaining_queries)}")
    write(intermediate_results_path, query_results)


    if not links_only:
        # group into webpage requests and pdf requests
        pdf_requests = []
        webpage_requests = []
        for results in tqdm(query_results, desc="Organizing retrieved urls for processing"):
            for result in results:
                url = result['link']
                
                # TODO: handle pdf and (assume) html parsing for now
                if is_pdf(result):
                    pdf_requests.append(url)
                else:
                    webpage_requests.append(url)

        # Validate request organization
        assert len(pdf_requests) + len(webpage_requests) == np.sum([len(results) for results in query_results])

        print(f"Num PDFs: {len(pdf_requests)}")
        print(f"Num webpages: {len(webpage_requests)}")

        method_path = os.path.join(full_session_name, "simple" if simple else "ours")

        webpage_url_to_raw = None
        if load_webpages:
            print("Building webpage url_to_raw...")
            # Handle webtext requests
            local_webpage_dir = os.path.join(intermediate_output_dir, method_path, "webpages", web_parse_mode)
            url_to_raw_path = os.path.join(local_webpage_dir, "url_to_raw.json")

            if not os.path.exists(local_webpage_dir):
                os.makedirs(local_webpage_dir, exist_ok=True)

                webpage_requests = list(set(webpage_requests))
                print(f"Num unique webpage urls: {len(webpage_requests)}")
                webpage_url_to_raw = asyncio.run(get_raw_texts_from_html(webpage_requests, web_parse_mode, max_concurrency))
                print(len(webpage_url_to_raw))
                write(url_to_raw_path, webpage_url_to_raw, mode='json')
            else: 
                webpage_url_to_raw = load_json(url_to_raw_path)
                print(len(webpage_url_to_raw))
        
        pdf_url_to_raw = None
        if load_pdfs:
            pdf_url_to_raw_path = os.path.join(intermediate_output_dir, method_path, "pdf_url_to_raw.json")
            if not os.path.exists(pdf_url_to_raw_path):
                print("Building pdf url_to_raw...")
                # Handle pdf requests asnd download raw content
                local_pdf_dir = os.path.join(intermediate_output_dir, full_session_name, "pdfs")
                url_to_out_path = os.path.join(local_pdf_dir, "url_to_out.json")
                if not os.path.exists(local_pdf_dir):
                    os.makedirs(local_pdf_dir, exist_ok=True)

                    pdf_requests = list(set(pdf_requests))
                    print(f"Num unique pdf urls: {len(pdf_requests)}")
                    url_to_out = asyncio.run(download_pdfs(pdf_requests, local_pdf_dir, max_concurrency))
                    write(url_to_out_path, url_to_out, mode='json')
                else: 
                    url_to_out = load_json(url_to_out_path)

                # Build a reverse map from out file_name to url
                url_to_out = {url : out for url, out in url_to_out.items() if out}
                out_to_url = {os.path.basename(urlparse(out).path) : url for url, out in url_to_out.items() if out}
                assert len(out_to_url) == len(url_to_out), f"{len(out_to_url)}, {len(url_to_out)}"

                pdf_url_to_raw = {}
                if simple:
                    # Failure count is given as second return value if needed
                    out_to_raw, _ = simple_pdf_extract(local_pdf_dir, max_workers)
                    
                    for out, raw in out_to_raw.items():
                        source_url = out_to_url[out]
                        assert source_url not in pdf_url_to_raw, source_url
                        pdf_url_to_raw[source_url] = raw
                else:
                    # Link downloaded pdfs to linearized versions
                    # All parsed pdfs through olmOCR should go under the following path
                    # TODO: Clear documentation needed to point users to using olmOCR for processing PDFs
                    parsed_pdf_dir = os.path.join(intermediate_output_dir, method_path, "parsed_pdfs")
                    if not os.path.isdir(parsed_pdf_dir):
                        logger.info("Raw PDFs are downloaded. Use olmOCR for linearizing and save parsed PDFs under ")

                    pdf_metadatas = extract_pdf_metadata(parsed_pdf_dir, max_workers)

                    for pdf_metadata in tqdm(pdf_metadatas, desc="Parsing pdfs..."):
                        source_file = pdf_metadata["metadata"]["Source-File"]
                        source_name = os.path.basename(urlparse(source_file).path)
                        assert source_name in out_to_url

                        source_url = out_to_url[source_name]
                        assert source_url not in pdf_url_to_raw, source_url
                        pdf_url_to_raw[source_url] = pdf_metadata["text"]

                    assert len(pdf_url_to_raw) == len(pdf_metadatas)
                write(pdf_url_to_raw_path, pdf_url_to_raw, mode='json')
            else:
                pdf_url_to_raw = load_json(pdf_url_to_raw_path)       
        
        print("Building web retrieval results...")
        total_results_with_raw = []
        bot_blocked = 0
        for i, results in tqdm(enumerate(query_results), desc="Applying retrieval results"):
            results_with_raw = []
            for result in results:
                # First check if url is in pdf url_to_raw if not None
                # Then check if available in webpage url_to_raw if not None
                result_link = result["link"]

                if pdf_url_to_raw and result_link in pdf_url_to_raw:
                    pdf_text = pdf_url_to_raw[result_link]
                    if len(pdf_text) == 0:
                        print("Empty pdf result??")
                    else:
                        results_with_raw.append({
                            "retrieval text": pdf_text,
                            "source": "olmOCR_pdf",
                            "result": result
                        })
                elif webpage_url_to_raw and result_link in webpage_url_to_raw:
                    retrieval_text = webpage_url_to_raw[result_link]
                    if len(retrieval_text) == 0:
                        print("Initially empty retrieval result??")
                    else:
                        retrieval_text = post_process_webpage(retrieval_text, result, simple)
                        
                        check = retrieval_text.lower()
                        if "cloudflare" in check or ("captcha" in check and "403" in check):
                            bot_blocked += 1 
                        else:
                            results_with_raw.append({
                                "retrieval text": retrieval_text,
                                "source": "webpage_parse",
                                "result": result
                            })                          

            total_results_with_raw.append(results_with_raw)

        
        out = [{"ctxs": results} | query for query, results in zip(queries, total_results_with_raw)]

        print(f"Num bot blocked: {bot_blocked}")
        print("writing retrieval results")
        # write out
        original_file_name, _ = os.path.splitext(os.path.basename(retrieval_queries))
        # original_dir_name = os.path.dirname(retrieval_queries)
        output_path = os.path.join(output_dir, method_path, web_parse_mode, f"{original_file_name}_web_retrieval_results.jsonl")
        write(output_path, out)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_queries', type=str, default=None, help='Path to JSONL file with each row containing a retrieval query and any additional metadata (AWS supported)')
    
    parser.add_argument('--retrieval_query_key', type=str, default='query', help='Key pointing to retrieval query in each JSON row')
    parser.add_argument('--k', type=int, default=10, help='Number of results to retrieve per query')    
    parser.add_argument('--num_tries', type=int, default=3, help='Number of tries for Google CSE search to fulfill all queries. Note this will increase api cost.')
    parser.add_argument('--session_name', type=str, help='Name to identify web retrieval session')
    parser.add_argument('--intermediate_output_dir', type=str, help='Output dir to place web retrieval links')

    parser.add_argument('--links_only', action='store_true', help='Whether to only retrieve and store web urls for queries')
    parser.add_argument('--load_pdfs', action='store_true', help='Whether to use olmOCR parsed pdf results to get result raw text')
    parser.add_argument('--load_webpages', action='store_true', help='Whether to useparsed webpage results to get result raw text')
    parser.add_argument('--web_parse_mode', default='default', help='Parsing strategy for web page urls')

    parser.add_argument('--simple', action='store_true', help='Whether to use simple parsing - borrowed from search-o1 repository')

    parser.add_argument('--max_concurrency', default=25, type=int, help='Maximum number of concurrent sessions for asynchronous applications.')
    parser.add_argument('--max_workers', default=6, type=int, help='Number of workers for use in multithreading applications')
    parser.add_argument('--output_dir', type=str, help='Output dir for JSONL matching input retrieval results as well as containing the top-k reranked results')

    # Testing params
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to randomly take from complete retrieval results')

    args = parser.parse_args()
    main(args)
