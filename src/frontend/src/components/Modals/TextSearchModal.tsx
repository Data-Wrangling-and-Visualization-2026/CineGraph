import React, { useState } from 'react';
import styles from './Modals.module.css';
import { searchMovies } from '../../api/graph';
import type { MovieData } from '../../types/movie';

interface TextSearchModalProps {
  onClose: () => void;
  onResults: (movies: MovieData[]) => void;
}

export function TextSearchModal({ onClose, onResults }: TextSearchModalProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await searchMovies({ description: query.trim() });
      onResults(response.movies);
      onClose(); 
    } catch (err: any) {
      setError(err.message || "Search error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <button className={styles.close_btn} onClick={onClose}>×</button>
        <h2 className={styles.title}>Semantic Search</h2>

        <form onSubmit={handleSearch}>
          <div className={styles.form_group}>
            <label>Describe movie mood or plot</label>
            <textarea 
              autoFocus
              className={styles.textarea} 
              value={query} 
              onChange={e => setQuery(e.target.value)}
              placeholder="e.g.: A group of teenagers goes to a cabin in the woods and encounters an ancient evil..."
            />
          </div>

          <button type="submit" className={styles.submit_btn} disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Find Movies'}
          </button>

          {error && <div className={styles.error_msg}>{error}</div>}
        </form>
      </div>
    </div>
  );
}