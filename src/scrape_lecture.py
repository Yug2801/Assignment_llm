import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import glob

base_url = "https://stanford-cs324.github.io/winter2022/lectures/"

def get_title_from_url(url):
    parsed_url = urlparse(url)
    path_components = parsed_url.path.split('/')
    if len(path_components) >= 3:
        return path_components[-2]
    return None

def scrape_lecture_notes(base_url):
    try:
        lecture_notes = []

        response = requests.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        main_content_div = soup.find('div', id='main-content')

        if main_content_div:
            lecture_links = main_content_div.find_all('a', href=True)

            output_folder = r"E:\Assignment2\data\lecture_notes"
            os.makedirs(output_folder, exist_ok=True)

            existing_files = glob.glob(os.path.join(output_folder, '*.txt'))
            for file in existing_files:
                os.remove(file)

            for idx, link in enumerate(lecture_links, start=1):
                lecture_url = urljoin(base_url, link['href'])
                lecture_title = get_title_from_url(lecture_url)

                lecture_response = requests.get(lecture_url)
                lecture_response.raise_for_status()

                lecture_soup = BeautifulSoup(lecture_response.content, 'html.parser')

                lecture_main_content = lecture_soup.find('div', id='main-content')

                if lecture_main_content:
                    lecture_content = lecture_main_content.text.strip()

                    filename = os.path.join(output_folder, f"{lecture_title}.txt")
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Lecture: {lecture_title}\n\n")
                        f.write(lecture_content)
                    
                    lecture_notes.append((lecture_title, lecture_content))
                else:
                    print(f"Warning: No main content found for lecture: {lecture_url}")
        else:
            print(f"Error: No main content div found on the page: {base_url}")

        return lecture_notes

    except Exception as e:
        print(f"Error fetching lecture notes: {e}")
        return []

lecture_notes = scrape_lecture_notes(base_url)

