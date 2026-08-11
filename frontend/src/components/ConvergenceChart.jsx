// shows accuracy and loss curves across all 10 federated rounds
import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getRounds } from '../api/client';

export default function ConvergenceChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    getRounds().then(res => {
      const formatted = res.data.map(r => ({
        round: r.round_num,
        accuracy: parseFloat(r.accuracy.toFixed(4)),
        loss: parseFloat(r.loss.toFixed(4)),
      }));
      setData(formatted);
    }).catch(console.error);
  }, []);

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Federated Training — Convergence</h2>
      <p style={styles.subtitle}>Accuracy and loss across 10 federated rounds (ε = ∞, no privacy)</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
          <XAxis dataKey="round" stroke="#888" label={{ value: 'Round', position: 'insideBottom', offset: -2, fill: '#888' }} />
          <YAxis stroke="#888" />
          <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #534AB7', color: '#fff' }} />
          <Legend />
          <Line type="monotone" dataKey="accuracy" stroke="#0F6E56" strokeWidth={2} dot={{ r: 4 }} name="Accuracy" />
          <Line type="monotone" dataKey="loss" stroke="#993C1D" strokeWidth={2} dot={{ r: 4 }} name="Loss" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles = {
  card: { background: '#12122a', borderRadius: 12, padding: 24, marginBottom: 24, border: '1px solid #2a2a3e' },
  title: { color: '#fff', fontSize: 18, fontWeight: 700, marginBottom: 4 },
  subtitle: { color: '#888', fontSize: 13, marginBottom: 20 },
};