import React, { useEffect, useState } from 'react';
import type { MyNode } from '../../types/graph';
import styles from './Sidebar.module.css';
import { fetchMovie } from '../../api/graph';
import { HexagonChart, type EmbeddingItem } from '../Charts/HexagonChart';

interface SidebarProps {
  selectedNode: MyNode | null;
  onClose: () => void;
}

export function Sidebar({ selectedNode, onClose }: SidebarProps) {
  const [embeddings, setEmbeddings] = useState<EmbeddingItem[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    // Если нода выбрана и у неё есть связанный ID фильма
    // Предполагается, что selectedNode.id или какое-то другое поле хранит ID фильма
    if (selectedNode?.id) {
      setLoading(true);
      fetchMovie(Number(selectedNode.id))
        .then(data => {
          setEmbeddings(data.embeddings);
        })
        .catch(err => {
          console.error("Failed to load movie data", err);
          setEmbeddings(null);
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [selectedNode]);

  if (!selectedNode) return null;

  return (
    <div className={styles.div_sidebar}>
      <button onClick={onClose} style={{ marginBottom: '20px', cursor: 'pointer' }}>
        Закрыть
      </button>
      
      <h2>{selectedNode.name}</h2>
      <p><b>Группа:</b> {selectedNode.group}</p>
      <p><b>Значимость:</b> {selectedNode.val}</p>

      <hr style={{ borderColor: '#444', margin: '20px 0' }} />
      
      <h3>Анализ эмбеддингов</h3>
      {loading ? (
        <p>Загрузка графика...</p>
      ) : embeddings ? (
        <HexagonChart embeddings={embeddings} />
      ) : (
        <p>Нет данных для графика</p>
      )}
    </div>
  );
}