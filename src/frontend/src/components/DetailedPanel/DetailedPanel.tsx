// src/components/DetailedPanel/DetailedPanel.tsx
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
  const { movieData, loading, engTitle, origTitle, showEngTitle } = useMovieData(isOpen ? node : null);

  if (!node) return null;
  const meta = movieData?.other_data;

  return (
    <div className={`${styles.panel} ${isOpen ? styles.panel_open : ''}`}>
      <button className={styles.close_button} onClick={onClose}>
        ← Back to Graph
      </button>

      {loading ? (
        <h2 style={{ color: '#aaa', marginTop: '20px' }}>Analyzing data...</h2>
      ) : movieData ? (
        <>
          {/* HEADER */}
          <div>
            <h1 style={{ fontSize: '2.5rem', margin: '0 0 10px 0' }}>{node.name}</h1>
            {meta?.tagline && (
              <p style={{ color: '#aaa', fontStyle: 'italic', fontSize: '1.2rem', margin: '0 0 10px 0' }}>
                "{meta.tagline}"
              </p>
            )}
          </div>

          <div className={styles.dashboard}>
            {/* LEFT COLUMN: CHARTS */}
            <div className={styles.left_column}>
              <EmbeddingsLineChart embeddings={movieData.embeddings} />

              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#ccc', textAlign: 'center' }}>Average Profile</h3>
                <HexagonChart embeddings={movieData.embeddings} />
              </div>
            </div>

            {/* RIGHT COLUMN: METADATA */}
            <div className={styles.right_column}>
              <h2 style={{ borderBottom: '1px solid #333', paddingBottom: '15px', marginTop: 0 }}>Movie Information</h2>
              
              {/* Overview */}
              {meta?.overview && (
                <div style={{ marginBottom: '20px', lineHeight: '1.6', color: '#ccc', fontSize: '14.5px', background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '8px' }}>
                  {meta.overview}
                </div>
              )}

              {/* Titles & Collection */}
              {origTitle && origTitle !== node.name && (
                <p style={{ margin: '10px 0', color: '#ddd' }}><b>Original Title:</b> {origTitle}</p>
              )}
              {showEngTitle && (
                <p style={{ margin: '10px 0', color: '#ddd' }}><b>English Title:</b> {engTitle}</p>
              )}
              {meta?.belongs_to_collection && (
                <p style={{ margin: '10px 0', color: '#4A90E2' }}><b>Collection:</b> {meta.belongs_to_collection.name}</p>
              )}

              <GenreTags genres={meta?.genres} />

              {/* Stats Grid */}
              <div className={styles.stats_grid}>
                <StatBox label="Release Date" value={meta?.release_date ? formatDate(meta.release_date) : movieData.year} />
                <StatBox 
                  label="Rating" 
                  value={meta?.vote_average ? `⭐ ${meta.vote_average.toFixed(1)} ${meta.vote_count ? `(${meta.vote_count} votes)` : ''}` : null} 
                />
                <StatBox label="Runtime" value={meta?.runtime ? `${meta.runtime} min.` : null} />
                <StatBox label="Popularity" value={meta?.popularity ? meta.popularity.toFixed(1) : null} />
                <StatBox label="Status" value={meta?.status || 'Released'} />
                <StatBox label="Budget" value={meta?.budget ? formatCurrency(meta.budget) : null} />
                <StatBox label="Revenue" value={meta?.revenue ? formatCurrency(meta.revenue) : null} />
              </div>

              {/* Countries, Companies, Languages */}
              <div style={{ marginTop: '25px', lineHeight: '1.8', fontSize: '14px' }}>
                
                {meta?.origin_country && meta.origin_country.length > 0 && (
                  <p><b style={{ color: '#888' }}>Origin Country: </b> {meta.origin_country.join(', ')}</p>
                )}

                {meta?.production_countries && meta.production_countries.length > 0 && (
                  <p><b style={{ color: '#888' }}>Production Countries: </b> {meta.production_countries.map(c => c.name).join(', ')}</p>
                )}

                {meta?.spoken_languages && meta.spoken_languages.length > 0 && (
                  <p><b style={{ color: '#888' }}>Spoken Languages: </b> {meta.spoken_languages.map(l => l.name).join(', ')}</p>
                )}

                {meta?.production_companies && meta.production_companies.length > 0 && (
                  <div style={{ marginTop: '10px' }}>
                    <b style={{ display: 'block', color: '#888', marginBottom: '4px' }}>Production Companies: </b> 
                    <p style={{ margin: 0 }}>{meta.production_companies.map(c => c.name).join(' • ')}</p>
                  </div>
                )}

                {/* Homepage Link */}
                {meta?.homepage && (
                  <p style={{ marginTop: '20px' }}>
                    <a 
                      href={meta.homepage} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      style={{ color: '#4A90E2', textDecoration: 'none', fontWeight: 'bold', display: 'inline-flex', alignItems: 'center', gap: '5px' }}
                    >
                      Official Website
                    </a>
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