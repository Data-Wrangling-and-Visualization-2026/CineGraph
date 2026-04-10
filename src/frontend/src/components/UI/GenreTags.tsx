import React from 'react';
import styles from './UI.module.css';

interface Genre {
  id: number;
  name: string;
}

interface GenreTagsProps {
  genres?: Genre[];
}

export function GenreTags({ genres }: GenreTagsProps) {
  if (!genres || genres.length === 0) return null;

  return (
    <div style={{ marginTop: '15px' }}>
      {genres.map(g => (
        <span key={g.id} className={styles.genre_tag}>{g.name}</span>
      ))}
    </div>
  );
}