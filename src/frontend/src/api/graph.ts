import type { MovieSubmission, SearchRequest, SearchMoviesResponse } from "../types/movie";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5555";

export async function fetchGraph(nodeId: number) {
  console.log("API URL:", API_URL);

  const res = await fetch(`${API_URL}/graph?node=${nodeId}`);

  if (!res.ok) {
    throw new Error("Failed to fetch graph");
  }

  return res.json();
}

export async function fetchMovie(id: number) {
  const res = await fetch(`${API_URL}/movie?id=${id}`);

  if (!res.ok) {
    throw new Error("Movie not found");
  }

  return res.json();
}

export async function addMovie(movieData: MovieSubmission) {
  const res = await fetch(`${API_URL}/add_movie`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(movieData),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Failed to add movie");
  }

  return res.json();
}

export async function searchMovies(payload: SearchRequest): Promise<SearchMoviesResponse> {
  const res = await fetch(`${API_URL}/search_movies`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Failed to search movies");
  }

  return res.json();
}