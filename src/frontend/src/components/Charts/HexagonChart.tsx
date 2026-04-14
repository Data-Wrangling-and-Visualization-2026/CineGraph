import React, { useMemo } from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip
} from 'recharts';

const EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

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

    const sum = [0, 0, 0, 0, 0, 0];

    embeddings.forEach(item => {
      item.embedding.forEach((val, i) => {
        sum[i] += val;
      });
    });

    const averages = sum.map(val => val / embeddings.length);
    const maxAvg = Math.max(...averages);

    return averages.map((avg, index) => {
      let scaledValue = 0;

      if (maxAvg > 0) {
        const normalized = avg / maxAvg;
        scaledValue = Math.pow(normalized, 0.5) * 10;
      }

      return {
        feature: EMOTIONS[index],
        value: Number(scaledValue.toFixed(2)),
        realValue: Number(avg.toFixed(4))
      };
    });
  }, [embeddings]);

  if (chartData.length === 0) {
    return <p style={{ color: '#aaa' }}>No data available</p>;
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
          tick={(props: any) => {
            const { x, y, cx, cy, payload } = props;

            const offset = 18;

            const dx = x - cx;
            const dy = y - cy;
            const length = Math.sqrt(dx * dx + dy * dy) || 1;

            const newX = x + (dx / length) * offset;
            const newY = y + (dy / length) * offset;

            return (
              <text
                x={newX}
                y={newY}
                textAnchor="middle"
                dominantBaseline="central"
                fill="#ccc"
                fontSize={13}
              >
                {payload.value.charAt(0).toUpperCase() + payload.value.slice(1)}
              </text>
            );
          }}
        />

        <PolarRadiusAxis
          angle={30}
          domain={[0, 10]}
          tickCount={6}
          tick={false}
          axisLine={false}
        />

        <Radar
          name="Value"
          dataKey="value"
          stroke="#00bfff"
          fill="#00bfff"
          fillOpacity={0.4}
          dot={{ r: 4, fill: '#00bfff' }}
          isAnimationActive={true}
        />

        <Tooltip
          formatter={(value: any, name: any, props: any) => [
            props.payload.realValue,
            'Value'
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