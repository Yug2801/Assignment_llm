import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

github_url = "https://github.com/Hannibal046/Awesome-LLM#milestone-papers"

def scrape_milestone_papers(github_url, save_path):
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an exception for bad response status

        soup = BeautifulSoup(response.content, 'html.parser')

        tables = soup.find_all('table')

        milestone_table = None
        for table in tables:
            if not table.has_attr('class') and not table.has_attr('id'):
                milestone_table = table
                break

        if milestone_table:
            df = pd.read_html(str(milestone_table))[0]

            df.columns = ['Date', 'Keywords', 'Institute', 'Paper', 'Publication']

            df = df.iloc[1:]

            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"Deleted existing file: {save_path}")

            output_folder = os.path.dirname(save_path)
            os.makedirs(output_folder, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Milestone papers table saved as {save_path}")

            return df
        else:
            print("Error: No suitable table found on the page.")
            return None

    except Exception as e:
        print(f"Error fetching milestone papers: {e}")
        return None

save_path = r"E:\Assignment2\data\milestone_papers.csv"
milestone_papers_df = scrape_milestone_papers(github_url, save_path)
if milestone_papers_df is not None:
    print("Milestone Papers Table:")
    print(milestone_papers_df.head())
