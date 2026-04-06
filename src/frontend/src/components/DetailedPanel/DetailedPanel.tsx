import React, { useEffect, useState, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import type { MyNode } from '../../types/graph';
import styles from './DetailedPanel.module.css';
import { fetchMovie } from '../../api/graph';
import { HexagonChart, type EmbeddingItem } from '../Charts/HexagonChart';

interface DetailedPanelProps {
  node: MyNode | null;
  isOpen: boolean;
  onClose: () => void;
}

interface MovieData {
  id: number;
  title: string;
  year: number;
  other_data: {
    title?: string;
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

// Форматтеры
const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(amount);
};

const formatDate = (dateString: string) => {
  if (!dateString) return '';
  const [year, month, day] = dateString.split('-');
  return `${day}.${month}.${year}`;
};

export function DetailedPanel({ node, isOpen, onClose }: DetailedPanelProps) {
  const [movieData, setMovieData] = useState<MovieData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Загружаем данные при открытии окна
  useEffect(() => {
    if (!isOpen || !node) return;

    const rawId = String(node.id);
    const movieId = Number(rawId.replace(/\D/g, ''));

    if (movieId && movieId > 0) {
      setLoading(true);
      fetchMovie(movieId)
        .then(data => setMovieData(data))
        .catch(err => {
          console.error("Ошибка загрузки данных в детальной панели:", err);
          setMovieData(null);
        })
        .finally(() => setLoading(false));
    }
  }, [node, isOpen]);

  // Трансформируем данные эмбеддингов для линейного графика (LineChart)
  const lineChartData = useMemo(() => {
    if (!movieData?.embeddings) return [];
    
    // Превращаем массив вида [{window_id: 0, embedding:[0.1, 0.2...]}] 
    // в[{ window: 0, F1: 0.1, F2: 0.2 ... }]
    return movieData.embeddings.map((item) => {
      const pointData: any = { window: item.window_id };
      item.embedding.forEach((val, i) => {
        pointData[`F${i + 1}`] = Number(val.toFixed(4));
      });
      return pointData;
    });
  }, [movieData]);

  // Цветовая палитра для 6 уникальных компонент
  const COMPONENT_COLORS =['#00bfff', '#ff4b4b', '#bd34fe', '#00fa9a', '#ffbf00', '#ff1493'];

  if (!node) return null;

  const meta = movieData?.other_data;
  const engTitle = (meta?.title || movieData?.title)?.trim();
  const origTitle = meta?.original_title?.trim();

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
              
              {/* Линейный график динамики эмбеддингов */}
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px' }}>
                <h3 style={{ margin: '0 0 20px 0', color: '#ccc' }}>Динамика компонент (Позиции)</h3>
                <div style={{ width: '100%', height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lineChartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="window" stroke="#888" tick={{ fontSize: 12 }} />
                      <YAxis stroke="#888" tick={{ fontSize: 12 }} />
                      
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e1e1e', border: '1px solid #444', borderRadius: '8px' }}
                        itemStyle={{ fontSize: '13px' }}
                        labelStyle={{ color: '#aaa', marginBottom: '5px' }}
                        formatter={(value: any, name: any) => [value, `Компонента ${name}`]}
                        labelFormatter={(label) => `Окно (позиция) ${label}`}
                      />
                      <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                      
                      {/* Отрисовываем 6 линий для каждой компоненты */}
                      {[1, 2, 3, 4, 5, 6].map((num, i) => (
                        <Line 
                          key={`F${num}`}
                          type="monotone" 
                          dataKey={`F${num}`} 
                          stroke={COMPONENT_COLORS[i]} 
                          strokeWidth={2}
                          dot={{ r: 2, fill: COMPONENT_COLORS[i], strokeWidth: 0 }}
                          activeDot={{ r: 5 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Шестиугольный профиль фильма */}
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px' }}>
                <h3 style={{ margin: '0 0 10px 0', color: '#ccc', textAlign: 'center' }}>Усредненный профиль</h3>
                <HexagonChart embeddings={movieData.embeddings} />
              </div>
            </div>

            {/* ПРАВАЯ КОЛОНКА: МЕТАДАННЫЕ */}
            <div className={styles.right_column}>
              <h2 style={{ borderBottom: '1px solid #333', paddingBottom: '15px', marginTop: 0 }}>Информация о фильме</h2>
              
              {/* Оригинальные названия */}
              {origTitle && origTitle !== node.name && (
                <p style={{ margin: '10px 0', color: '#ddd' }}><b>Ориг. название:</b> {origTitle}</p>
              )}
              {meta?.original_language && meta.original_language !== 'en' && engTitle && engTitle !== node.name && engTitle !== origTitle && (
                <p style={{ margin: '10px 0', color: '#ddd' }}><b>Англ. название:</b> {engTitle}</p>
              )}

              {/* Жанры (Красивыми тегами) */}
              {meta?.genres && meta.genres.length > 0 && (
                <div style={{ marginTop: '15px' }}>
                  {meta.genres.map(g => (
                    <span key={g.id} className={styles.genre_tag}>{g.name}</span>
                  ))}
                </div>
              )}

              {/* Сетка статистик */}
              <div className={styles.stats_grid}>
                <div className={styles.stat_box}>
                  <div className={styles.stat_label}>Дата релиза</div>
                  <p className={styles.stat_value}>{meta?.release_date ? formatDate(meta.release_date) : movieData.year}</p>
                </div>

                <div className={styles.stat_box}>
                  <div className={styles.stat_label}>Оценка</div>
                  <p className={styles.stat_value}>
                    {meta?.vote_average ? `⭐ ${meta.vote_average.toFixed(1)}` : 'Нет оценки'}
                  </p>
                </div>

                <div className={styles.stat_box}>
                  <div className={styles.stat_label}>Длительность</div>
                  <p className={styles.stat_value}>
                    {meta?.runtime ? `${meta.runtime} мин.` : 'Неизвестно'}
                  </p>
                </div>

                <div className={styles.stat_box}>
                  <div className={styles.stat_label}>Статус</div>
                  <p className={styles.stat_value}>{meta?.status || 'Выпущен'}</p>
                </div>

                {meta?.budget !== undefined && meta.budget > 0 && (
                  <div className={styles.stat_box}>
                    <div className={styles.stat_label}>Бюджет</div>
                    <p className={styles.stat_value}>{formatCurrency(meta.budget)}</p>
                  </div>
                )}

                {meta?.revenue !== undefined && meta.revenue > 0 && (
                  <div className={styles.stat_box}>
                    <div className={styles.stat_label}>Сборы</div>
                    <p className={styles.stat_value}>{formatCurrency(meta.revenue)}</p>
                  </div>
                )}
              </div>

              {/* Страны и Компании */}
              <div style={{ marginTop: '25px', lineHeight: '1.6' }}>
                {meta?.production_countries && meta.production_countries.length > 0 && (
                  <p>
                    <b style={{ color: '#888' }}>Страна производства: </b> 
                    {meta.production_countries.map(c => c.name).join(', ')}
                  </p>
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