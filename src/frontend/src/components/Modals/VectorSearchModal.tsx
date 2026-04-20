import React, { useState } from 'react';
import styles from './Modals.module.css';
import { searchMovies } from '../../api/graph';
import type { MovieData } from '../../types/movie';

interface VectorSearchModalProps {
  onClose: () => void;
  onResults: (movies: MovieData[]) => void;
}

const EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

const INITIAL_STATE = {
  start: [0, 0, 0, 0, 0, 0],
  middle: [0, 0, 0, 0, 0, 0],
  end: [0, 0, 0, 0, 0, 0],
};

function calculateVariance(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
  return variance;
}

export function VectorSearchModal({ onClose, onResults }: VectorSearchModalProps) {
  const [vectors, setVectors] = useState(INITIAL_STATE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (phase: 'start' | 'middle' | 'end', index: number, value: string) => {
    setVectors(prev => {
      const newPhase = [...prev[phase]];
      newPhase[index] = parseFloat(value);
      return { ...prev, [phase]: newPhase };
    });
  };

  const handleSearch = async () => {
    setLoading(true);
    setError(null);

    const variance = [0, 1, 2, 3, 4, 5].map(i => {
      const valuesForComponent = [vectors.start[i], vectors.middle[i], vectors.end[i]];
      return calculateVariance(valuesForComponent);
    });

    const finalVector = [
      ...vectors.start,
      ...vectors.middle,
      ...vectors.end,
      ...variance
    ];

    try {
      const response = await searchMovies({ description: finalVector });
      onResults(response.movies);
      onClose();
    } catch (err: any) {
      setError(err.message || "Vector search error");
    } finally {
      setLoading(false);
    }
  };

  const renderColumn = (title: string, phase: 'start' | 'middle' | 'end') => (
    <div className={styles.sliders_column}>
      <h4>{title}</h4>
      {[0, 1, 2, 3, 4, 5].map(i => (
        <div key={i} className={styles.slider_container}>
          <div className={styles.slider_header}>
            <span style={{ textTransform: 'capitalize' }}>{EMOTIONS[i]}</span>
            <span className={styles.slider_value}>{vectors[phase][i].toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            min="0" max="1" step="0.01" 
            className={styles.range_input}
            value={vectors[phase][i]}
            onChange={(e) => handleChange(phase, i, e.target.value)}
          />
        </div>
      ))}
    </div>
  );

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} style={{ width: '700px' }} onClick={(e) => e.stopPropagation()}>
        <button className={styles.close_btn} onClick={onClose}>×</button>
        <h2 className={styles.title}>Embedding Search</h2>

        <div className={styles.sliders_grid}>
          {renderColumn('Opening', 'start')}
          {renderColumn('Middle', 'middle')}
          {renderColumn('Ending', 'end')}
        </div>

        <button onClick={handleSearch} className={styles.submit_btn} disabled={loading}>
          {loading ? 'Analyzing...' : 'Find Similar Movies'}
        </button>

        {error && <div className={styles.error_msg}>{error}</div>}
      </div>
    </div>
  );
}