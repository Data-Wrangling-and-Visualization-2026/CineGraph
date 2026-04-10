import React from 'react';
import styles from './UI.module.css';

interface StatBoxProps {
  label: string;
  value: React.ReactNode;
}

export function StatBox({ label, value }: StatBoxProps) {
  if (!value || value === '0' || value === '$0') return null; // Не рендерим пустые значения

  return (
    <div className={styles.stat_box}>
      <div className={styles.stat_label}>{label}</div>
      <p className={styles.stat_value}>{value}</p>
    </div>
  );
}