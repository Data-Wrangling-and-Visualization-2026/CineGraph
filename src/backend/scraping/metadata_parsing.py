import wikipedia
import json
import re
from bs4 import BeautifulSoup


def clean_wiki_text(text):
    """
    Final cleanup: handles currency symbols, spacing,
    and parentheses formatting.
    """
    if not text:
        return None

    # Remove citations [1], [a], etc.
    text = re.sub(r'\[.*?\]', '', text)

    # Normalize dashes and non-breaking spaces
    text = text.replace('\xa0', ' ').replace('\u2013', '-').replace('\u2014', '-')

    # Fix spacing inside parentheses: "( $ 10 )" -> "($10)"
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\$\s+', '$', text)

    # Collapse multiple commas and surrounding whitespace
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'(,\s*)+', ', ', text)

    # Collapse all whitespace into single spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip().strip(',').strip()


def get_film_metadata(film_name):
    try:
        search_results = wikipedia.search(f"{film_name} film")
        if not search_results:
            return {"error": "No results"}

        page = wikipedia.page(search_results[0], auto_suggest=False)
        soup = BeautifulSoup(page.html(), 'html.parser')
        infobox = soup.find('table', {'class': re.compile(r'infobox.*vevent|infobox')})

        metadata = {
            "Title": page.title,
            "Director": None, "Country": None, "Running time": None,
            "Budget": None, "Box office": None
        }

        if infobox:
            for row in infobox.find_all('tr'):
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    label = th.get_text(" ", strip=True).lower()

                    # Target specific tags for replacement to avoid merging words
                    # but only if they act as block elements.
                    for tag in td.find_all(['br', 'p', 'li']):
                        tag.insert_after(', ')

                    value = td.get_text(" ", strip=True)

                    if "directed by" in label:
                        metadata["Director"] = clean_wiki_text(value)
                    elif "country" in label or "countries" in label:
                        metadata["Country"] = clean_wiki_text(value)
                    elif "running time" in label:
                        metadata["Running time"] = clean_wiki_text(value)
                    elif "budget" in label:
                        metadata["Budget"] = clean_wiki_text(value)
                    elif "box office" in label:
                        metadata["Box office"] = clean_wiki_text(value)

        return metadata

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    test_films = [
        "The Godfather", "Avengers Endgame", "Parasite", "12 Angry Men",
        "Spirited Away", "The Blair Witch Project", "Monty Python Holy Grail",
        "Lord of the Rings Two Towers", "Mad Max Fury Road", "Dune Part Two"
    ]

    for film in test_films:
        data = get_film_metadata(film)
        print(json.dumps(data, indent=4, ensure_ascii=False))