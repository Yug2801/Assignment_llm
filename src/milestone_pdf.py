import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader

github_url = "https://github.com/Hannibal046/Awesome-LLM#milestone-papers"

def scrape_milestone_papers(github_url, save_folder):
    try:
        response = requests.get(github_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        tables = soup.find_all('table')

        milestone_table = None
        for table in tables:
            if not table.has_attr('class') and not table.has_attr('id'):
                milestone_table = table
                break

        if milestone_table is None:
            raise Exception("No suitable table found.")

        df = pd.read_html(str(milestone_table))[0]
        df.columns = ['Date', 'Keywords', 'Institute', 'Paper', 'Publication']
        df = df.iloc[1:]

        os.makedirs(save_folder, exist_ok=True)
        for filename in os.listdir(save_folder):
            file_path = os.path.join(save_folder, filename)
            try:
                if filename.endswith(".txt"):
                    os.remove(file_path)
                    print(f"Deleted existing file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        all_links = milestone_table.find_all('a', href=True)

        valid_href_values = []
        for link in all_links:
            if 'target' in link.attrs and link['target'] == '_blank':
                continue
            valid_href_values.append(link['href'])

        print("All links found in the milestone papers table:")
        print(valid_href_values)

        for idx, link in enumerate(valid_href_values):
            pdf_text = extract_pdf_text(link)

            keyword = df.iloc[idx]['Keywords']
            if not keyword:
                keyword = f"pdf{idx + 1}"

            txt_filename = f"{keyword}.txt"
            txt_filepath = os.path.join(save_folder, txt_filename)
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(pdf_text)
            print(f"Saved text from PDF {idx + 1} as '{txt_filename}'")

        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request exception occurred: {e}")
    except Exception as e:
        print(f"Error fetching milestone papers: {e}")

    return None

def extract_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        temp_pdf_path = 'temp.pdf'
        with open(temp_pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        pdf_text = ''
        with open(temp_pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

        os.remove(temp_pdf_path)

        return pdf_text.strip()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while downloading PDF from {pdf_url}: {e}")
        return ''
    except requests.exceptions.RequestException as e:
        print(f"Request exception occurred while downloading PDF from {pdf_url}: {e}")
        return ''
    except Exception as e:
        print(f"Error extracting PDF text from {pdf_url}: {e}")
        return ''

# Example usage
save_folder = r"E:\Assignment2\data\milestone_papers_text"
milestone_papers_df = scrape_milestone_papers(github_url, save_folder)
if milestone_papers_df is not None:
    print("Milestone Papers Table:")
    print(milestone_papers_df.head())
