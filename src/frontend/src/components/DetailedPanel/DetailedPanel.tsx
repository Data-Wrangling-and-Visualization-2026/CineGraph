import React from 'react';
import styles from './DetailedPanel.module.css';
import type { MyNode } from '../../types/graph';
import { useMovieData } from '../../hooks/useMovieData';
import { formatCurrency, formatDate } from '../../utils/formatters';

import { HexagonChart } from '../Charts/HexagonChart';
import { EmbeddingsLineChart } from '../Charts/EmbeddingsLineChart';
import { StatBox } from '../UI/StatBox';
import { GenreTags } from '../UI/GenreTags';

interface DetailedPanelProps {
  node: MyNode | null;
  isOpen: boolean;
  onClose: () => void;
}

export function DetailedPanel({ node, isOpen, onClose }: DetailedPanelProps) {
  // МАГИЯ ЗДЕСЬ: Вся сложная логика спрятана в хук!
  const { movieData, loading, engTitle, origTitle, showEngTitle } = useMovieData(isOpen ? node : null);

  if (!node) return null;
  const meta = movieData?.other_data;

  return (
    <div className={`${styles.panel} ${isOpen ? styles.panel_open : ''}`}>
      <button className={styles.close_button} onClick={onClose}>
        ← Назад к графу
      </button>

      {loading ? (
        <h2 style={{ color: '#aaa', marginTop: '20px' }}>Анализ данных...</h2>
      ) : movieData ? (
        <>
          {/* ЗАГОЛОВОК */}
          <div>
            <h1 style={{ fontSize: '2.5rem', margin: '0 0 10px 0' }}>{node.name}</h1>
            {meta?.tagline && (
              <p style={{ color: '#aaa', fontStyle: 'italic', fontSize: '1.2rem', margin: '0 0 10px 0' }}>
                «{meta.tagline}»
              </p>
            )}
          </div>

          <div className={styles.dashboard}>
            {/* ЛЕВАЯ КОЛОНКА: ГРАФИКИ */}
            <div className={styles.left_column}>
              <EmbeddingsLineChart embeddings={movieData.embeddings} />

              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#ccc', textAlign: 'center' }}>Усредненный профиль</h3>
                <HexagonChart embeddings={movieData.embeddings} />
              </div>
            </div>

            {/* ПРАВАЯ КОЛОНКА: МЕТАДАННЫЕ */}
            <div className={styles.right_column}>
              <h2 style={{ borderBottom: '1px solid #333', paddingBottom: '15px', marginTop: 0 }}>Информация о фильме</h2>
              
              {origTitle && origTitle !== node.name && (
                <p style={{ margin: '10px 0', color: '#ddd' }}><b>Ориг. название:</b> {origTitle}</p>
              )}
              {showEngTitle && (
                <p style={{ margin: '10px 0', color: '#ddd' }}><b>Англ. название:</b> {engTitle}</p>
              )}

              <GenreTags genres={meta?.genres} />

              {/* Сетка статистик с помощью нового переиспользуемого компонента */}
              <div className={styles.stats_grid}>
                <StatBox label="Дата релиза" value={meta?.release_date ? formatDate(meta.release_date) : movieData.year} />
                <StatBox label="Оценка" value={meta?.vote_average ? `⭐ ${meta.vote_average.toFixed(1)}` : null} />
                <StatBox label="Длительность" value={meta?.runtime ? `${meta.runtime} мин.` : null} />
                <StatBox label="Статус" value={meta?.status || 'Выпущен'} />
                <StatBox label="Бюджет" value={meta?.budget ? formatCurrency(meta.budget) : null} />
                <StatBox label="Сборы" value={meta?.revenue ? formatCurrency(meta.revenue) : null} />
              </div>

              {/* Страны и Компании */}
              <div style={{ marginTop: '25px', lineHeight: '1.6' }}>
                {meta?.production_countries && meta.production_countries.length > 0 && (
                  <p><b style={{ color: '#888' }}>Страна производства: </b> {meta.production_countries.map(c => c.name).join(', ')}</p>
                )}
                {meta?.production_companies && meta.production_companies.length > 0 && (
                  <p style={{ marginTop: '10px' }}>
                    <b style={{ display: 'block', color: '#888', marginBottom: '4px' }}>Кинокомпании: </b> 
                    {meta.production_companies.map(c => c.name).join(' • ')}
                  </p>
                )}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}