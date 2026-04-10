import { useState, useEffect, useMemo } from 'react';
import { fetchMovie } from '../api/graph';
import type { MovieData } from '../types/movie';
import type { MyNode } from '../types/graph';

export function useMovieData(node: MyNode | null) {
  const [movieData, setMovieData] = useState<MovieData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!node) {
      setMovieData(null);
      return;
    }

    const rawId = String(node.id);
    const movieId = Number(rawId.replace(/\D/g, ''));

    if (movieId && movieId > 0) {
      setLoading(true);
      setError(null);
      fetchMovie(movieId)
        .then(data => setMovieData(data))
        .catch(err => {
          console.error("Ошибка загрузки данных фильма:", err);
          setError("Не удалось загрузить данные");
          setMovieData(null);
        })
        .finally(() => setLoading(false));
    } else {
      setMovieData(null);
    }
  }, [node]);

  // Вычисляем логику отображения названий прямо здесь
  const titles = useMemo(() => {
    if (!node || !movieData) {
      return { engTitle: null, origTitle: null, showOrigTitle: false, showEngTitle: false };
    }

    const meta = movieData.other_data;
    const nodeName = node.name?.trim();
    const origTitle = meta?.original_title?.trim();
    const engTitle = (meta?.title || movieData?.title)?.trim();

    const isSameName = (a?: string, b?: string) => {
      if (!a || !b) return false;
      return a.toLowerCase() === b.toLowerCase();
    };

    const showOrigTitle = !!origTitle && !isSameName(origTitle, nodeName);
    
    const showEngTitle = !!(
      meta?.original_language && 
      meta.original_language !== 'en' && 
      engTitle && 
      !isSameName(engTitle, nodeName) && 
      !isSameName(engTitle, origTitle)
    );

    return { engTitle, origTitle, showOrigTitle, showEngTitle };
  }, [node, movieData]);

  return { movieData, loading, error, ...titles };
}