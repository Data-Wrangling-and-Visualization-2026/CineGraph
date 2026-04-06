import React, { useEffect, useState } from 'react';
import type { MyNode } from '../../types/graph';
import styles from './Sidebar.module.css';
import { fetchMovie } from '../../api/graph';
import { HexagonChart, type EmbeddingItem } from '../Charts/HexagonChart';

interface SidebarProps {
  selectedNode: MyNode | null;
  onClose: () => void;
  onOpenDetails: () => void;
  isHidden?: boolean;
}

interface MovieData {
  id: number;
  title: string; 
  year: number;
  other_data: {
    title?: string; // ДОБАВЛЕНО: берем title прямо из JSON-объекта
    genres?: { id: number; name: string }[];
    runtime?: number;
    release_date?: string;
    vote_average?: number;
    vote_count?: number;
    production_countries?: { name: string }[];
    production_companies?: { name: string }[];
    status?: string;
    tagline?: string;
    budget?: number;
    revenue?: number;
    original_title?: string;     
    original_language?: string;  
  } | null;
  embeddings: EmbeddingItem[];
}

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(amount);
};

const formatDate = (dateString: string) => {
  if (!dateString) return '';
  const[year, month, day] = dateString.split('-');
  return `${day}.${month}.${year}`;
};

export function Sidebar({ selectedNode, onClose, onOpenDetails, isHidden }: SidebarProps) {
  const [movieData, setMovieData] = useState<MovieData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!selectedNode) return;

    const rawId = String(selectedNode.id);
    const numericString = rawId.replace(/\D/g, ''); 
    const movieId = Number(numericString);

    if (movieId && movieId > 0) {
      setLoading(true);
      fetchMovie(movieId)
        .then(data => setMovieData(data))
        .catch(err => {
          console.error("Ошибка загрузки данных фильма:", err);
          setMovieData(null);
        })
        .finally(() => setLoading(false));
    } else {
      console.warn("Не удалось извлечь числовой ID из:", selectedNode.id);
      setMovieData(null);
    }
  }, [selectedNode]);

  if (!selectedNode) return null;

  const meta = movieData?.other_data;

  // --- ЛОГИКА ОТОБРАЖЕНИЯ НАЗВАНИЙ ---
  const nodeName = selectedNode.name?.trim();
  const origTitle = meta?.original_title?.trim();
  // Отдаем приоритет title из other_data (Там обычно лежит качественный английский перевод)
  const engTitle = (meta?.title || movieData?.title)?.trim();

  // Функция для безопасного сравнения строк (без учета регистра)
  const isSameName = (a?: string, b?: string) => {
    if (!a || !b) return false;
    return a.toLowerCase() === b.toLowerCase();
  };

  // Показывать оригинальное название, если оно есть и не совпадает с H2 (Именем ноды)
  const showOrigTitle = origTitle && !isSameName(origTitle, nodeName);

  // Показывать английское название, если:
  // 1. Язык оригинала не английский
  // 2. Оно существует
  // 3. Оно не совпадает с именем ноды в графе
  // 4. Оно не совпадает с оригинальным названием (чтобы не было дублей)
  const showEngTitle = meta?.original_language && 
                       meta.original_language !== 'en' && 
                       engTitle && 
                       !isSameName(engTitle, nodeName) && 
                       !isSameName(engTitle, origTitle);

  return (
    <div className={`${styles.div_sidebar} ${isHidden ? styles.hidden : ''}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
        <button onClick={onClose} style={{ cursor: 'pointer', padding: '5px 10px' }}>
          Закрыть
        </button>
        
        {/* Кнопка открытия большой панели */}
        <button 
          onClick={onOpenDetails} 
          style={{ cursor: 'pointer', padding: '5px 10px', background: '#00bfff', color: '#fff', border: 'none', borderRadius: '4px' }}
        >
          Анализ
        </button>
      </div>
      
      {/* Главное название (Имя узла) */}
      <h2 style={{ marginBottom: '5px' }}>{selectedNode.name}</h2>
      
      {meta?.tagline && (
        <p style={{ margin: '0 0 15px 0', fontStyle: 'italic', color: '#aaa', fontSize: '13px' }}>
          «{meta.tagline}»
        </p>
      )}
      
      <div style={{ marginBottom: '15px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
        <p style={{ margin: '3px 0', fontSize: '13px' }}><b>Группа графа:</b> {selectedNode.group}</p>
        <p style={{ margin: '3px 0', fontSize: '13px' }}><b>Значимость:</b> {selectedNode.val}</p>
      </div>

      <hr style={{ borderColor: '#444', margin: '15px 0' }} />

      <h3>О фильме</h3>
      {loading ? (
        <p style={{ color: '#aaa', fontSize: '14px' }}>Загрузка информации...</p>
      ) : movieData ? (
        <div style={{ fontSize: '14px', lineHeight: '1.6', color: '#ddd' }}>
          
          {/* Динамическое отображение дополнительных названий */}
          {showOrigTitle && (
            <p style={{ margin: '5px 0' }}><b>Ориг. название:</b> {origTitle}</p>
          )}

          {showEngTitle && (
            <p style={{ margin: '5px 0' }}><b>Англ. название:</b> {engTitle}</p>
          )}
          
          <p style={{ margin: '5px 0' }}><b>Год:</b> {movieData.year}</p>
          
          {meta ? (
            <>
              {meta.release_date && (
                <p style={{ margin: '5px 0' }}><b>Дата релиза:</b> {formatDate(meta.release_date)}</p>
              )}

              {meta.genres && meta.genres.length > 0 && (
                <p style={{ margin: '5px 0' }}>
                  <b>Жанры:</b> {meta.genres.map(g => g.name).join(', ')}
                </p>
              )}
              
              {meta.production_countries && meta.production_countries.length > 0 && (
                <p style={{ margin: '5px 0' }}>
                  <b>Страна:</b> {meta.production_countries.map(c => c.name).join(', ')}
                </p>
              )}

              {meta.runtime !== undefined && meta.runtime > 0 && (
                <p style={{ margin: '5px 0' }}><b>Длительность:</b> {meta.runtime} мин.</p>
              )}

              {meta.vote_average !== undefined && meta.vote_average > 0 && (
                <p style={{ margin: '5px 0' }}>
                  <b>Оценка:</b> ⭐ {meta.vote_average.toFixed(1)} / 10 
                  {meta.vote_count ? ` (${meta.vote_count} чел.)` : ''}
                </p>
              )}

              {meta.budget !== undefined && meta.budget > 0 && (
                <p style={{ margin: '5px 0' }}><b>Бюджет:</b> {formatCurrency(meta.budget)}</p>
              )}
              {meta.revenue !== undefined && meta.revenue > 0 && (
                <p style={{ margin: '5px 0' }}><b>Сборы:</b> {formatCurrency(meta.revenue)}</p>
              )}

              {meta.production_companies && meta.production_companies.length > 0 && (
                <div style={{ margin: '10px 0' }}>
                  <b>Студии:</b>
                  <p style={{ margin: '2px 0', fontSize: '12px', color: '#bbb' }}>
                    {meta.production_companies.map(c => c.name).join(' • ')}
                  </p>
                </div>
              )}
            </>
          ) : (
            <p style={{ color: '#999', fontStyle: 'italic' }}>Подробная информация отсутствует</p>
          )}
        </div>
      ) : (
        <p style={{ color: '#aaa' }}>Данные не найдены</p>
      )}

      <hr style={{ borderColor: '#444', margin: '20px 0' }} />
      
      <h3 style={{ marginBottom: '5px' }}>Анализ характеристик</h3>
      {loading ? (
        <p style={{ color: '#aaa' }}>Загрузка графика...</p>
      ) : movieData?.embeddings ? (
        <HexagonChart embeddings={movieData.embeddings} />
      ) : null}
      
    </div>
  );
}   