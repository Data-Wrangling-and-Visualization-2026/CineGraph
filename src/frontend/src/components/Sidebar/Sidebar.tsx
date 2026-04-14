import React from 'react';
import type { MyNode } from '../../types/graph';
import styles from './Sidebar.module.css';
import { useMovieData } from '../../hooks/useMovieData';
import { HexagonChart } from '../Charts/HexagonChart';
import { SidebarMovieInfo } from './SidebarMovieInfo';

interface SidebarProps {
  selectedNode: MyNode | null;
  onClose: () => void;
  onOpenDetails: () => void;
  isHidden?: boolean;
}

export function Sidebar({ selectedNode, onClose, onOpenDetails, isHidden }: SidebarProps) {
  const { movieData, loading, error, origTitle, engTitle, showOrigTitle, showEngTitle } = useMovieData(selectedNode);

  if (!selectedNode) return null;

  const meta = movieData?.other_data;

  return (
    <div className={`${styles.div_sidebar} ${isHidden ? styles.hidden : ''}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
        <button onClick={onClose} style={{ cursor: 'pointer', padding: '5px 10px' }}>
          Close
        </button>
        
        <button 
          onClick={onOpenDetails} 
          style={{ cursor: 'pointer', padding: '5px 10px', background: '#00bfff', color: '#fff', border: 'none', borderRadius: '4px' }}
        >
          Analysis
        </button>
      </div>
      
      <h2 style={{ marginBottom: '5px' }}>{selectedNode.name}</h2>
      
      {meta?.tagline && (
        <p style={{ margin: '0 0 15px 0', fontStyle: 'italic', color: '#aaa', fontSize: '13px' }}>
          "{meta.tagline}"
        </p>
      )}
      

      <hr style={{ borderColor: '#444', margin: '15px 0' }} />

      <h3>About Movie</h3>
      {loading ? (
        <p style={{ color: '#aaa', fontSize: '14px' }}>Loading info...</p>
      ) : error ? (
        <p style={{ color: '#ff4b4b', fontSize: '14px' }}>{error}</p>
      ) : movieData ? (
        <SidebarMovieInfo 
          movieData={movieData}
          origTitle={origTitle}
          engTitle={engTitle}
          showOrigTitle={showOrigTitle}
          showEngTitle={showEngTitle}
        />
      ) : (
        <p style={{ color: '#aaa' }}>Data not found</p>
      )}

      <hr style={{ borderColor: '#444', margin: '20px 0' }} />
      
      <h3 style={{ marginBottom: '5px' }}>Feature Analysis</h3>
      {loading ? (
        <p style={{ color: '#aaa' }}>Loading chart...</p>
      ) : movieData?.embeddings ? (
        <HexagonChart embeddings={movieData.embeddings} />
      ) : null}
      
    </div>
  );
}