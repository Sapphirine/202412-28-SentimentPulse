from bs4 import BeautifulSoup
import http.client
import json

# API key removed from public repo
API_KEY = ''
# Get HTML of a website using a RapidAPI API
# Code provided by RapidAPI
def get_article_html(url):
    conn = http.client.HTTPSConnection("scrapeninja.p.rapidapi.com")
    payload = {"url": url}
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': "scrapeninja.p.rapidapi.com",
        'Content-Type': "application/json"
    }
    conn.request("POST", "/scrape", json.dumps(payload), headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

# Get article text from the url
def get_article_text(url):
    # Get the html using the previous function
    html = get_article_html(url)
    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    # Find the container with article text using CSS selectors
    article_container = soup.select_one("div[class*='caas-body'], div[class*='body']")
    # If desired seelctor is there, continue
    if article_container:
        # Extract all paragraph texts within the found container
        paragraphs = article_container.find_all('p')
        article_text = []
        # Get text by paragraph
        for para in paragraphs:
            # Get the raw text from the paragraph
            para_text = para.get_text()
            # Remove any text from hyperlinks within the paragraph
            links = para.find_all('a')
            for link in links:
                link_text = link.get_text()
                para_text = para_text.replace(link_text, "")  # Exclude link text
            # Add cleaned paragraph text to the article content
            article_text.append(para_text.strip())
        # Join all paragraphs into a single string
        return '\n'.join(article_text)
    else:
        print(f"No article content found")
        return None
