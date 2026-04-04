import React, { useMemo } from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip
} from 'recharts';

export interface EmbeddingItem {
  window_id: number;
  embedding: number[];
}

interface HexagonChartProps {
  embeddings: EmbeddingItem[];
}

export function HexagonChart({ embeddings }: HexagonChartProps) {
  const chartData = useMemo(() => {
    if (!embeddings || embeddings.length === 0) return [];

    const sum =[0, 0, 0, 0, 0, 0];
    
    // Суммируем
    embeddings.forEach(item => {
      item.embedding.forEach((val, i) => {
        sum[i] += val;
      });
    });

    // 1. Находим средние значения
    const averages = sum.map(val => val / embeddings.length);
    
    // 2. Находим самое большое значение из 6-ти у этого конкретного фильма
    const maxAvg = Math.max(...averages);

    return averages.map((avg, index) => {
      let scaledValue = 0;
      
      if (maxAvg > 0) {
        // а) Относительная нормализация: самый длинный луч всегда будет равен 1
        const normalized = avg / maxAvg; 
        
        // б) Извлекаем квадратный корень, чтобы визуально "вытянуть" слишком маленькие значения из центра
        // в) Умножаем на 10 для шкалы графика
        scaledValue = Math.pow(normalized, 0.5) * 10;
      }
      
      return {
        feature: `F${index + 1}`,
        value: Number(scaledValue.toFixed(2)), // Значение для отрисовки (от 0 до 10)
        realValue: Number(avg.toFixed(4))      // Настоящее математическое значение для подсказки
      };
    });
  },[embeddings]);

  if (chartData.length === 0) {
    return <p style={{ color: '#aaa' }}>Нет данных для графика</p>;
  }

  return (
    <div style={{ display: 'flex', justifyContent: 'center', marginTop: '10px' }}>
      <RadarChart 
        cx="50%" 
        cy="50%" 
        outerRadius={80} 
        width={260} 
        height={260} 
        data={chartData}
      >
        <PolarGrid stroke="#555" />
        
        <PolarAngleAxis 
          dataKey="feature" 
          tick={{ fill: '#ccc', fontSize: 13 }} 
        />
        
        {/* ИСПРАВЛЕНИЕ СЕТКИ: tickCount={6} заставит Recharts нарисовать кольца с идеально равным шагом */}
        <PolarRadiusAxis 
          angle={30} 
          domain={[0, 10]} 
          tickCount={6} 
          tick={false} 
          axisLine={false} 
        />
        
        <Radar
          name="Значение"
          dataKey="value"
          stroke="#00bfff"
          fill="#00bfff"
          fillOpacity={0.4}
          dot={{ r: 4, fill: '#00bfff' }}
          isAnimationActive={true}
        />

        <Tooltip 
          // ИСПРАВЛЕНИЕ ПОДСКАЗКИ: Рисуем красивые масштабы, но при наведении показываем "realValue"
          formatter={(value: any, name: any, props: any) => [
            props.payload.realValue, 
            'Значение'
          ]}
          contentStyle={{ 
            backgroundColor: '#2e2c2c', 
            border: '1px solid #555', 
            borderRadius: '8px',
            color: '#fff' 
          }}
          itemStyle={{ color: '#00bfff' }}
        />
      </RadarChart>
    </div>
  );
}