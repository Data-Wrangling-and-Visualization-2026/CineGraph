// src/components/Charts/PlotlyChart.tsx
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

interface PlotlyChartProps {
  apiEndpoint: string;
}

export function PlotlyChart({ apiEndpoint }: PlotlyChartProps) {
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    
    // Запрашиваем JSON с графиком у бэкенда
    fetch(apiEndpoint)
      .then(res => {
        if (!res.ok) throw new Error('Failed to load chart data');
        return res.json();
      })
      .then(data => {
        // Если Python вернул строку (иногда to_json() возвращает string), парсим её
        const parsedData = typeof data === 'string' ? JSON.parse(data) : data;
        setChartData(parsedData);
      })
      .catch(err => {
        console.error("Chart fetch error:", err);
        setError(err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [apiEndpoint]);

  if (loading) {
    return (
      <div style={{ color: '#888', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        Loading interactive chart...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ color: '#ff4b4b', textAlign: 'center' }}>
        <p>Error loading chart.</p>
        <p style={{ fontSize: '12px' }}>{error}</p>
        <p style={{ fontSize: '12px', color: '#555' }}>Endpoint: {apiEndpoint}</p>
      </div>
    );
  }

  if (!chartData) return null;

  return (
    <Plot
      data={chartData.data}
      layout={{
        ...chartData.layout,
        autosize: true, // Авто-ресайз под контейнер
        paper_bgcolor: 'transparent', // Делаем фон прозрачным, чтобы сливалось с нашим сайтом
        plot_bgcolor: 'transparent',
      }}
      useResizeHandler={true}
      style={{ width: '100%', height: '100%', minHeight: '500px' }}
      config={{ 
        responsive: true, 
        displayModeBar: false // Убираем верхнюю панель инструментов Plotly (по желанию можешь поставить true)
      }}
    />
  );
}