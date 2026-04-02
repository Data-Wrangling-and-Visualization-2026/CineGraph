import React, { useMemo } from 'react';

// Типы на основе вашего JSON
export interface EmbeddingItem {
  window_id: number;
  embedding: number[];
}

interface HexagonChartProps {
  embeddings: EmbeddingItem[];
}

export function HexagonChart({ embeddings }: HexagonChartProps) {
  // 1. Вычисляем среднее значение для каждой из 6 позиций
  const avgEmbedding = useMemo(() => {
    if (!embeddings || embeddings.length === 0) return[0, 0, 0, 0, 0, 0];
    
    const sum =[0, 0, 0, 0, 0, 0];
    embeddings.forEach(item => {
      item.embedding.forEach((val, i) => {
        sum[i] += val;
      });
    });
    
    return sum.map(val => val / embeddings.length);
  }, [embeddings]);

  // 2. Скалируем значения от 0 до 10
  // Если исходные данные[0..1], просто умножаем на 10.
  // (Если вам нужна строгая min-max нормализация, формула была бы: (val - min)/(max - min) * 10)
  const scaledValues = avgEmbedding.map(v => Math.min(Math.max(v * 10, 0), 10));

  // Настройки SVG холста
  const size = 260; // общий размер
  const center = size / 2;
  const radius = 90; // максимальный радиус (соответствует значению 10)

  // Функция перевода (значение, индекс) -> координаты (x, y)
  const getPointCoordinates = (value: number, index: number) => {
    // 6 углов, начинаем сверху (-Math.PI / 2)
    const angle = (Math.PI * 2 * index) / 6 - Math.PI / 2; 
    const r = (value / 10) * radius; // радиус пропорционален значению
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle)
    };
  };

  // Координаты для полигона графика
  const chartPoints = scaledValues.map((val, i) => getPointCoordinates(val, i));
  const polygonPointsString = chartPoints.map(p => `${p.x},${p.y}`).join(' ');

  // Уровни фоновой сетки (отметки 2, 4, 6, 8, 10)
  const gridLevels =[2, 4, 6, 8, 10];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        
        {/* Оси (линии из центра к краям) */}
        {[0, 1, 2, 3, 4, 5].map(i => {
          const { x, y } = getPointCoordinates(10, i);
          return (
            <line key={`axis-${i}`} x1={center} y1={center} x2={x} y2={y} stroke="#555" strokeWidth="1" />
          );
        })}

        {/* Фоновая паутина (сетка) */}
        {gridLevels.map(level => {
          const levelPoints =[0, 1, 2, 3, 4, 5]
            .map(i => getPointCoordinates(level, i))
            .map(p => `${p.x},${p.y}`)
            .join(' ');
          return (
            <polygon key={`grid-${level}`} points={levelPoints} fill="none" stroke="#444" strokeWidth="1" />
          );
        })}

        {/* Сам график: заливка и контур */}
        <polygon 
          points={polygonPointsString} 
          fill="rgba(0, 191, 255, 0.3)" 
          stroke="#00bfff" 
          strokeWidth="2" 
        />

        {/* Точки (узлы) на краях графика */}
        {chartPoints.map((p, i) => (
          <circle key={`dot-${i}`} cx={p.x} cy={p.y} r={4} fill="#00bfff" />
        ))}

        {/* Подписи осей */}
        {[0, 1, 2, 3, 4, 5].map(i => {
          const { x, y } = getPointCoordinates(12, i); // выносим текст чуть за радиус
          return (
            <text 
              key={`label-${i}`} 
              x={x} y={y} 
              fontSize="12" 
              fill="#ccc" 
              textAnchor="middle" 
              dominantBaseline="middle"
            >
              F{i + 1}: {scaledValues[i].toFixed(1)}
            </text>
          );
        })}
      </svg>
    </div>
  );
}