import React from 'react';
import type { MovieData } from '../../types/movie';
import { formatCurrency, formatDate } from '../../utils/formatters';

interface SidebarMovieInfoProps {
  movieData: MovieData;
  origTitle: string | null | undefined;
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
        <p style={{ margin: '5px 0' }}><b>Original Title:</b> {origTitle}</p>
      )}

      {showEngTitle && (
        <p style={{ margin: '5px 0' }}><b>English Title:</b> {engTitle}</p>
      )}
      
      <p style={{ margin: '5px 0' }}><b>Year:</b> {movieData.year}</p>
      
      {meta ? (
        <>
          {meta.release_date && (
            <p style={{ margin: '5px 0' }}><b>Release Date:</b> {formatDate(meta.release_date)}</p>
          )}

          {meta.genres && meta.genres.length > 0 && (
            <p style={{ margin: '5px 0' }}>
              <b>Genres:</b> {meta.genres.map(g => g.name).join(', ')}
            </p>
          )}
          
          {meta.production_countries && meta.production_countries.length > 0 && (
            <p style={{ margin: '5px 0' }}>
              <b>Country:</b> {meta.production_countries.map(c => c.name).join(', ')}
            </p>
          )}

          {meta.runtime !== undefined && meta.runtime > 0 && (
            <p style={{ margin: '5px 0' }}><b>Runtime:</b> {meta.runtime} min</p>
          )}

          {meta.vote_average !== undefined && meta.vote_average > 0 && (
            <p style={{ margin: '5px 0' }}>
              <b>Rating:</b> ⭐ {meta.vote_average.toFixed(1)} / 10 
              {meta.vote_count ? ` (${meta.vote_count} votes)` : ''}
            </p>
          )}

          {meta.budget !== undefined && meta.budget > 0 && (
            <p style={{ margin: '5px 0' }}><b>Budget:</b> {formatCurrency(meta.budget)}</p>
          )}
          {meta.revenue !== undefined && meta.revenue > 0 && (
            <p style={{ margin: '5px 0' }}><b>Revenue:</b> {formatCurrency(meta.revenue)}</p>
          )}

          {meta.production_companies && meta.production_companies.length > 0 && (
            <div style={{ margin: '10px 0' }}>
              <b>Studios:</b>
              <p style={{ margin: '2px 0', fontSize: '12px', color: '#bbb' }}>
                {meta.production_companies.map(c => c.name).join(' • ')}
              </p>
            </div>
          )}
        </>
      ) : (
        <p style={{ color: '#999', fontStyle: 'italic' }}>Detailed information unavailable</p>
      )}
    </div>
  );
}