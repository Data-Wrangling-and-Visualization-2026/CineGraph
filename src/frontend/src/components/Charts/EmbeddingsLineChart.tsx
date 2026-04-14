import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { EmbeddingItem } from '../../types/movie';

const COMPONENT_COLORS = ['#00bfff', '#ff4b4b', '#bd34fe', '#00fa9a', '#ffbf00', '#ff1493'];
const EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

interface EmbeddingsLineChartProps {
  embeddings?: EmbeddingItem[];
}

export function EmbeddingsLineChart({ embeddings }: EmbeddingsLineChartProps) {
  const lineChartData = useMemo(() => {
    if (!embeddings) return [];
    
    return embeddings.map((item) => {
      const pointData: any = { window: item.window_id };
      item.embedding.forEach((val, i) => {
        if (EMOTIONS[i]) {
          pointData[EMOTIONS[i]] = Number(val.toFixed(4));
        }
      });
      return pointData;
    });
  }, [embeddings]);

  if (!embeddings || embeddings.length === 0) return null;

  return (
    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px' }}>
      <h3 style={{ margin: '0 0 20px 0', color: '#ccc' }}>Emotion analysis</h3>
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
              formatter={(value: any, name: any) => [
                value, 
                typeof name === 'string' ? name.charAt(0).toUpperCase() + name.slice(1) : name
              ]}
              labelFormatter={(label) => `Window (position) ${label}`}
            />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            
            {EMOTIONS.map((emotion, i) => (
              <Line 
                key={emotion}
                type="monotone" 
                dataKey={emotion} 
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
  );
}