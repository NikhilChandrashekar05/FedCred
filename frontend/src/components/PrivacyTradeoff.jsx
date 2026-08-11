// privacy-utility tradeoff chart — accuracy vs epsilon
import { useEffect, useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';
import { getPrivacyTradeoff } from '../api/client';

export default function PrivacyTradeoff() {
  const [data, setData] = useState([]);

  useEffect(() => {
    getPrivacyTradeoff().then(res => {
      const formatted = res.data
        .filter(d => d.accuracy > 0)
        .map(d => ({
          epsilon: d.epsilon === 1e308 ? 999 : d.epsilon,
          accuracy: parseFloat((d.accuracy * 100).toFixed(1)),
          label: d.epsilon === 1e308 ? '∞ (no DP)' : `ε=${d.epsilon}`,
        }));
      setData(formatted);
    }).catch(console.error);
  }, []);

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Privacy-Utility Tradeoff</h2>
      <p style={styles.subtitle}>How differential privacy (Opacus) affects model accuracy. Lower ε = stronger privacy = more noise.</p>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
          <XAxis dataKey="epsilon" stroke="#888" name="Epsilon" type="number">
            <Label value="Epsilon (ε)" position="insideBottom" offset={-2} fill="#888" />
          </XAxis>
          <YAxis dataKey="accuracy" stroke="#888" name="Accuracy" unit="%" />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #534AB7', color: '#fff' }}
            formatter={(value, name) => [name === 'accuracy' ? `${value}%` : value, name]}
          />
          <Scatter data={data} fill="#534AB7" />
        </ScatterChart>
      </ResponsiveContainer>
      <div style={styles.legend}>
        <span style={styles.legendItem}>🔵 Each point = one training run at that epsilon level</span>
        <span style={styles.legendItem}>ε = ∞ shown as 999 on axis (no privacy, baseline)</span>
      </div>
    </div>
  );
}

const styles = {
  card: { background: '#12122a', borderRadius: 12, padding: 24, marginBottom: 24, border: '1px solid #2a2a3e' },
  title: { color: '#fff', fontSize: 18, fontWeight: 700, marginBottom: 4 },
  subtitle: { color: '#888', fontSize: 13, marginBottom: 20 },
  legend: { display: 'flex', flexDirection: 'column', gap: 4, marginTop: 12 },
  legendItem: { color: '#666', fontSize: 12 },
};