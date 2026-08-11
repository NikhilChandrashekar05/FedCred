import ConvergenceChart from './components/ConvergenceChart';
import TrainingDashboard from './components/TrainingDashboard';
import PrivacyTradeoff from './components/PrivacyTradeoff';
import LoanForm from './components/LoanForm';

export default function App() {
  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <h1 style={styles.headerTitle}>FedCredit</h1>
        <p style={styles.headerSub}>Federated Learning Credit Risk Dashboard</p>
      </div>

      <div style={styles.container}>
        <div style={styles.statsRow}>
          <StatCard label="Banks" value="3" color="#0F6E56" />
          <StatCard label="Training Rounds" value="10" color="#534AB7" />
          <StatCard label="Total Borrowers" value="1.26M" color="#185FA5" />
          <StatCard label="Privacy" value="Opacus DP" color="#854F0B" />
        </div>

        <TrainingDashboard />
        <ConvergenceChart />
        <PrivacyTradeoff />
        <LoanForm />
      </div>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{ ...styles.statCard, borderTop: `3px solid ${color}` }}>
      <div style={{ ...styles.statValue, color }}>{value}</div>
      <div style={styles.statLabel}>{label}</div>
    </div>
  );
}

const styles = {
  app: { minHeight: '100vh', background: '#0d0d1f', fontFamily: "'Inter', sans-serif" },
  header: { background: '#12122a', borderBottom: '1px solid #2a2a3e', padding: '24px 40px' },
  headerTitle: { color: '#fff', fontSize: 28, fontWeight: 800, margin: 0 },
  headerSub: { color: '#888', fontSize: 14, margin: '4px 0 0' },
  container: { maxWidth: 1100, margin: '0 auto', padding: '32px 24px' },
  statsRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 },
  statCard: { background: '#12122a', borderRadius: 10, padding: '20px 24px', border: '1px solid #2a2a3e' },
  statValue: { fontSize: 24, fontWeight: 800, marginBottom: 4 },
  statLabel: { color: '#888', fontSize: 13 },
};