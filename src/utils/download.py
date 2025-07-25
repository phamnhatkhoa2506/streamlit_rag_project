import os
import wget
import json
import ssl
import urllib3
import re

from typing import Dict

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

pattern = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"


def sanitize_filename(filename: str) -> str:
    """
        Sanitize filename by removing or replacing invalid characters
    """

    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)

    # Remove any leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')

    return sanitized


def is_exist(file_link: str, dir_name: str) -> bool:
    '''
        Check if the file already exists
    '''

    sanitized_title = sanitize_filename(file_link["title"])
    return os.path.exists(f"data_source/{dir_name}/{sanitized_title}.pdf")


def load_links_file(links_file_path: str) -> Dict[str, str]:
    with open(links_file_path, "r", encoding="utf-8") as f:
        file_links = json.load(f)

    return file_links


def download(dir_name: str, links: Dict[str, str]) -> None:
    os.makedirs(f"data_source", exist_ok=True)
    os.makedirs(f"data_source/{dir_name}", exist_ok=True)

    if not all(re.match(pattern, link["url"]) for link in links):
        raise ValueError("Url isn't invalid")
    
    for link in links:
        if not is_exist(link, dir_name):
            sanitized_title = sanitize_filename(link["title"])
            wget.download(link["url"], out=f"data_source/{dir_name}/{sanitized_title}.pdf")
    

def save_documents(dir_name: str, files: list[any]) -> None:
    os.makedirs(f"data_source", exist_ok=True)
    os.makedirs(f"data_source/{dir_name}", exist_ok=True)

    for file in files:
        file_path = os.path.join(f"data_source/{dir_name}/{file.name}")
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


if __name__ == '__main__':
    links_file = load_links_file('./links.json')
    print(links_file)

    download(links_file)