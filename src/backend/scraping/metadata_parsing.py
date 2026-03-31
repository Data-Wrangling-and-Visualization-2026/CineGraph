import requests
import json
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '..', '.env')

load_dotenv(env_path)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is not set. Please check your .env file.")


def get_film_metadata(film_name):
    # TMDB API Base URLs
    search_url = "https://api.themoviedb.org/3/search/movie"
    details_url = "https://api.themoviedb.org/3/movie/"

    try:
        # Search for the movie to get its TMDB ID
        search_params = {
            "api_key": TMDB_API_KEY,
            "query": film_name
        }
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()

        # Check if we got any results
        if not search_data.get("results"):
            return {"error": f"No results found for '{film_name}'"}

        # Extract the ID of the first (most relevant) search result
        movie_id = search_data["results"][0]["id"]

        # Fetch the top-level details using the movie ID
        details_response = requests.get(
            f"{details_url}{movie_id}",
            params={"api_key": TMDB_API_KEY}
        )
        details_response.raise_for_status()
        details_data = details_response.json()

        # Filter the response to only include your desired fields
        desired_fields = [
            "budget", "genres", "origin_country", "original_language",
            "original_title", "popularity", "production_companies",
            "production_countries", "release_date", "revenue",
            "runtime", "spoken_languages", "status", "tagline",
            "title", "vote_average", "vote_count"
        ]

        # Build the final dictionary
        metadata = {field: details_data.get(field) for field in desired_fields}
        return metadata

    except requests.exceptions.RequestException as e:
        return {"error": f"API Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    test_films = [
        "The Godfather", "Avengers Endgame", "Parasite", "12 Angry Men",
        "Spirited Away", "The Blair Witch Project", "Monty Python Holy Grail",
        "Lord of the Rings Two Towers", "Mad Max Fury Road", "Dune Part Two"
    ]

    for film in test_films:
        data = get_film_metadata(film)
        print(json.dumps(data, indent=4, ensure_ascii=False))