from api.base_models import MovieSubmission

def validate_movie(movie: MovieSubmission) -> bool:
    """
    Basic validation check for the submitted movie.
    """
    if not movie.title or not movie.title.strip() or not movie.year:
        raise ValueError("Movie title cannot be empty.")

    if movie.year < 1888 or movie.year > 2100:
        raise ValueError("Invalid movie year.")

    if not movie.subtitles or not movie.subtitles.strip():
        raise ValueError("Subtitles text cannot be empty.")

    return True