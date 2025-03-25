import os
import tarfile
import requests


def extract_data(url: str, tar_path: str, extract_path: str) -> None:
    response = requests.get(url, stream=True)
    with open(tar_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)
            