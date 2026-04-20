import React from 'react';
import styles from './Toolbar.module.css';

interface ToolbarProps {
  onOpenAddMovie: () => void;
  onOpenTextSearch: () => void;
  onOpenVectorSearch: () => void;
}

export function Toolbar({ onOpenAddMovie, onOpenTextSearch, onOpenVectorSearch }: ToolbarProps) {
  const iconSize = { width: '30px', height: '30px' };

  return (
    <div className={styles.toolbar}>
      <button 
        className={styles.icon_button} 
        onClick={onOpenAddMovie} 
        title="Add Movie"
        style={{ width: 'auto', height: 'auto', padding: '10px' }}
      >
        <svg viewBox="0 0 24 24" style={iconSize} fill="currentColor">
          <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
        </svg>
      </button>

      <button 
        className={styles.icon_button} 
        onClick={onOpenTextSearch} 
        title="Text Search"
        style={{ width: 'auto', height: 'auto', padding: '10px' }}
      >
        <svg viewBox="0 0 24 24" style={iconSize} fill="currentColor">
          <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" />
          <text 
            x="9.5" 
            y="10.2" 
            fontSize="7" 
            fontWeight="900" 
            fill="currentColor" 
            textAnchor="middle" 
            dominantBaseline="middle"
          >
            T
          </text>
        </svg>
      </button>

      <button 
        className={styles.icon_button} 
        onClick={onOpenVectorSearch} 
        title="Vector Search"
        style={{ width: 'auto', height: 'auto', padding: '10px' }}
      >
        <svg viewBox="0 0 24 24" style={iconSize} fill="currentColor">
          <path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z" />
        </svg>
      </button>
    </div>
  );
}