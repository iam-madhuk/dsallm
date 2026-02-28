import requests
from bs4 import BeautifulSoup
import os

def scrape_dsa_content(url, output_file):
    """
    Specialized scraper for DSA content.
    Attempts to identify problem statements and solutions.
    """
    print(f"Scraping DSA content from {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()

        # Try to find common DSA article containers (GeeksforGeeks, etc.)
        content = soup.find('article') or soup.find('div', class_='entry-content') or soup.body
        
        text = content.get_text(separator='\n')

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text_content = "\n".join(lines)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
            
        print(f"Saved DSA content to {output_file}")
        return True
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return False

if __name__ == "__main__":
    import sys
    os.makedirs("data", exist_ok=True)
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.geeksforgeeks.org/binary-search/"
    scrape_dsa_content(target_url, "data/dsa_raw.txt")
