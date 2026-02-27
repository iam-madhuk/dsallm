import requests
from bs4 import BeautifulSoup
import os

def scrape_url(url, output_file):
    print(f"Scraping content from {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text_content = "\n".join(chunk for chunk in chunks if chunk)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
            
        print(f"Saved scraped content to {output_file}")
        return True
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scrape_url(sys.argv[1], "data/raw_scraped.txt")
    else:
        print("Usage: python scraper.py <url>")
