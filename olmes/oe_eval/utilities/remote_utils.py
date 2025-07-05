import logging
import os
from typing import List
from urllib.parse import urlparse

import boto3
import smart_open
from tqdm import tqdm
from urllib.parse import urlparse
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)

S3_CACHE_DIR = os.environ.get("S3_CACHE_DIR") or os.path.expanduser("~/.cache/oe_eval_s3_cache")


def cache_s3_folder(s3_path, cache_dir=None):
    cache_dir = cache_dir or S3_CACHE_DIR
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    s3_folder = parsed_url.path.lstrip("/")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    files: List[str] = []
    s3_filenames = []
    for obj in bucket.objects.filter(Prefix=s3_folder):
        s3_filenames.append(obj.key)
    logger.info(f"Downloading {len(s3_filenames)} files from {s3_path} to {cache_dir}")
    for s3_filename in s3_filenames:
        local_filename = os.path.join(cache_dir, s3_filename)
        files.append(local_filename)
        if os.path.exists(local_filename):
            continue
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        bucket.download_file(s3_filename, local_filename)
    local_dir = sorted([os.path.dirname(file) for file in files], key=len)[0]
    logger.info(f"Finished downloading to {local_dir}")
    return {"local_dir": local_dir, "files": files}


def parse_s3_url(s3_url):
    """
    Parse an S3 URL into bucket and key.
    """
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    return parsed.netloc, parsed.path.lstrip('/')


def download_file_from_s3(s3_url, destination_file=None):
    """
    Download a file from an S3 URL.
    """
    bucket_name, object_name = parse_s3_url(s3_url)
    if destination_file is None:
        destination_file = object_name.split('/')[-1]  # Just the filename

    s3_client = boto3.client('s3')

    try:
        s3_client.download_file(bucket_name, object_name, destination_file)
        print(f"Downloaded '{s3_url}' to '{destination_file}'")
    except ClientError as e:
        print(f"Download error: {e}")
    except NoCredentialsError:
        print("AWS credentials not available.")


def upload_directory(local_dir: str, remote_dir: str):
    local_paths = [
        os.path.join(root, post_fn) for root, _, files in os.walk(local_dir) for post_fn in files
    ]
    dest_paths = [
        f"{remote_dir.rstrip('/')}/{os.path.relpath(local_path, local_dir).lstrip('/')}"
        for local_path in local_paths
    ]
    it = tqdm(
        iterable=zip(local_paths, dest_paths),
        total=len(local_paths),
        desc=f"Uploading files from {local_dir} to {remote_dir}",
    )
    for local_path, dest_path in it:
        with smart_open.open(local_path, "rb") as f, smart_open.open(dest_path, "wb") as g:
            g.write(f.read())
