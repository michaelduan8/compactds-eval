import os
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.utils import load_json


def get_leaf_files(directory):
    leaf_files = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  # No subdirectories in this folder → it's a leaf folder
            leaf_files.extend([os.path.join(root, file) for file in files])
    return leaf_files


def build_dclm_overlap_map(dclm_overlap_dir):
    def _get_overlap_metadata(file_path):
        temp_url_to_raw = {}#defaultdict(set)
        dclm_overlap_metadata = load_json(file_path)
        for url, texts in dclm_overlap_metadata.items():
            temp_url_to_raw[url] = random.choice(texts) #.update(texts)

        return temp_url_to_raw
    
    dclm_url_to_raw_list = []
    dclm_overlap_metadata_paths = get_leaf_files(dclm_overlap_dir)
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_file = {executor.submit(_get_overlap_metadata, file): file for file in dclm_overlap_metadata_paths}

        # Wrap the as_completed iterator with tqdm for progress tracking
        for future in tqdm(as_completed(future_to_file), total=len(dclm_overlap_metadata_paths), desc="Processing JSON files"):
            dclm_url_to_raw_list.append(future.result())

    dclm_overlap_url_to_raw = {}
    for dclm_url_to_raw in dclm_url_to_raw_list:
        for url, texts in dclm_url_to_raw.items():
            #dclm_overlap_url_to_raw[url].update(texts)
            dclm_overlap_url_to_raw[url] = texts#[0]

    return dclm_overlap_url_to_raw

# dclm_url_to_raw = None
# if load_dclm_overlap:
#     dclm_url_to_raw_path = os.path.join(intermediate_output_dir, method_path, "dclm_url_to_raw.json")
#     if not os.path.exists(dclm_url_to_raw_path):
#         print("Building dclm overlap url_to_raw...")
#         # All dclm_overlap metadata should go under the following path
#         dclm_overlap_dir = os.path.join(intermediate_output_dir, method_path, "dclm_overlap")
#         assert os.path.isdir(dclm_overlap_dir)

#         dclm_url_to_raw = build_dclm_overlap_map(dclm_overlap_dir)
#         dclm_url_to_raw = {url: text for url, text in dclm_url_to_raw.items()}
#         write(dclm_url_to_raw_path, dclm_url_to_raw, mode='json')
#         print(f"Number of urls with dclm overlap: {len(dclm_url_to_raw)}")
#     else:
#         dclm_url_to_raw = load_json(dclm_url_to_raw_path)