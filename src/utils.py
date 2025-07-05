import boto3
import csv
import json
import os
import pickle
import smart_open
import boto3

from urllib.parse import urlparse
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm

def list_s3_objects_in_directory(bucket_name, prefix, suffix=None):
    """
    List all objects in a specific directory (prefix) in an S3 bucket.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    all_file_keys = []
    
    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for i, page in enumerate(pages):
            if 'Contents' not in page:
                print(f"No objects found in '{prefix}' within bucket '{bucket_name}' for page {i}.")
            else:
                for obj in page['Contents']:
                    path = obj['Key']
                    if not suffix or (suffix and path.endswith(suffix)):
                        all_file_keys.append(path)

    except Exception as e:
        print(f"Error listing objects in bucket {bucket_name} with prefix '{prefix}': {e}")

    return all_file_keys

def stream_file_from_s3(bucket_name, file_key, tp={}, head=None):
    """
    Stream a file from S3 and return its content.
    """
    lines = []
    url = f"s3://{bucket_name}/{file_key}"
    with smart_open.open(url, transport_params=tp) as f_in:
        for line in f_in:
            lines.append(json.loads(line))

            if head and len(lines) >= head:
                break
    
    return lines

def load_json(file_path):
    return json.load(smart_open.open(file_path, 'r'))

def load_jsonl(file_path):
    with smart_open.open(file_path, 'r') as f_in:
        return [json.loads(line) for line in tqdm(f_in)] #tqdm
    
def write(file_name, contents, mode='jsonl'):
    if 's3' not in file_name:
        # If writing locally, make sure output directory exists
        output_dir = os.path.dirname(file_name)
        os.makedirs(output_dir, exist_ok=True)

    if mode == 'jsonl':
        with smart_open.open(file_name, 'w') as f:
            for content in tqdm(contents):
                f.write(f"{json.dumps(content)}\n")
    elif mode == 'json':
        with smart_open.open(file_name, 'w') as f:
            json.dump(contents, f, indent=4)
    elif mode == 'csv':
        with smart_open.open(file_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=contents[0].keys())
            writer.writeheader()  # Write the header (keys of the dictionary)
            writer.writerows(contents)  # Write the rows (dictionaries)
    elif mode == 'pkl':
        with smart_open.open(file_name, 'wb') as f:
            pickle.dump(contents, f, protocol=pickle.HIGHEST_PROTOCOL)


def sort_dicts_by_key(data, sort_keys, reversed=False):
    """
    Sort a list of dictionaries by a specified key.
    If the key is not present in a dictionary, it will be placed at the end.
    """
    if not isinstance(sort_keys, list):
        sort_keys = [sort_keys]
    
    with_key = [item for item in data if all([key in item for key in sort_keys])]
    without_key = [item for item in data if any([key not in item for key in sort_keys])]

    # Sort only the ones with the key
    with_key_sorted = sorted(with_key, key=lambda x: [float(x[key]) for key in sort_keys], reverse=reversed)
    # Combine back
    return with_key_sorted + without_key, len(with_key)