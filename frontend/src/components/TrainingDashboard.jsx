// live feed of training rounds — polls every 5 seconds
import { useEffect, useState } from 'react';
import { getRounds, getStatus } from '../api/client';

export default function TrainingDashboard() {
  const [rounds, setRounds] = useState([]);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const fetchData = () => {
      getRounds().then(res => setRounds(res.data)).catch(console.error);
      getStatus().then(res => setStatus(res.data)).catch(console.error);
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Training Status</h2>

      {status && (
        <div style={styles.statusRow}>
          <span style={{ ...styles.badge, background: status.status === 'complete' ? '#0F6E56' : '#534AB7' }}>
            {status.status.toUpperCase()}
          </span>
          <span style={styles.statusText}>
            Round {status.current_round} / {status.total_rounds} — Latest accuracy: {(status.latest_accuracy * 100).toFixed(1)}%
          </span>
        </div>
      )}

      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              {['Round', 'Loss', 'Accuracy', 'Clients'].map(h => (
                <th key={h} style={styles.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...rounds].reverse().map(r => (
              <tr key={r.round_num} style={styles.tr}>
                <td style={styles.td}>{r.round_num}</td>
                <td style={styles.td}>{r.loss.toFixed(4)}</td>
                <td style={{ ...styles.td, color: '#0F6E56', fontWeight: 600 }}>{(r.accuracy * 100).toFixed(1)}%</td>
                <td style={styles.td}>{r.num_clients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const styles = {
  card: { background: '#141435', borderRadius: 12, padding: 24, marginBottom: 24, border: '1px solid #2a2a3e' },
  title: { color: '#fff', fontSize: 18, fontWeight: 700, marginBottom: 16 },
  statusRow: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 },
  badge: { padding: '4px 12px', borderRadius: 20, color: '#fff', fontSize: 12, fontWeight: 700 },
  statusText: { color: '#888', fontSize: 13 },
  tableWrap: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { color: '#888', fontSize: 12, fontWeight: 600, textAlign: 'left', padding: '8px 12px', borderBottom: '1px solid #2a2a3e' },
  tr: { borderBottom: '1px solid #1a1a2e' },
  td: { color: '#ccc', fontSize: 13, padding: '8px 12px' },
};