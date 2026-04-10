import React from 'react';
import type { MovieData } from '../../types/movie';
import { formatCurrency, formatDate } from '../../utils/formatters';

interface SidebarMovieInfoProps {
  movieData: MovieData;
  origTitle: string | null;
  engTitle: string | null;
  showOrigTitle: boolean;
  showEngTitle: boolean;
}

export function SidebarMovieInfo({ 
  movieData, origTitle, engTitle, showOrigTitle, showEngTitle 
}: SidebarMovieInfoProps) {
  const meta = movieData.other_data;

  return (
    <div style={{ fontSize: '14px', lineHeight: '1.6', color: '#ddd' }}>
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
  );
}