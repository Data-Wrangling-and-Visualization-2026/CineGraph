export interface EmbeddingItem {
  window_id: number;
  embedding: number[];
}

export interface MovieData {
  id: number;
  title: string;
  year: number;
  other_data: {
    title?: string;
    genres?: { id: number; name: string }[];
    runtime?: number;
    release_date?: string;
    vote_average?: number;
    vote_count?: number;
    production_countries?: { name: string }[];
    production_companies?: { name: string }[];
    status?: string;
    tagline?: string;
    budget?: number;
    revenue?: number;
    original_title?: string;
    original_language?: string;
  } | null;
  embeddings: EmbeddingItem[];
}

export interface MovieSubmission {
  title: string;
  year: number;
  subtitles: string; // Будем отправлять пустую строку
  other_data: {
    budget?: number;
    overview?: string;
    genres?: { id: number; name: string }[];
    release_date?: string;
    runtime?: number;
    status?: string;
    vote_average?: number;
    vote_count?: number;
  };
}

export interface SearchRequest {
  description: string | number[]; // Либо текст, либо массив из 24 float
}

export interface SearchMoviesResponse {
  movies: MovieData[]; // Бэкенд возвращает список фильмов
}