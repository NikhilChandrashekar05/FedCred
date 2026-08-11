// loan application form — hits /predict/ and shows approve/deny + reasons
import { useState } from 'react';
import { predictLoan } from '../api/client';

const defaultValues = {
  loan_amnt: 15000, int_rate: 12.5, installment: 350, annual_inc: 55000,
  dti: 18.5, fico_range_low: 680, fico_range_high: 684, inq_last_6mths: 1,
  open_acc: 8, pub_rec: 0, revol_bal: 12000, revol_util: 45.0,
  total_acc: 15, delinq_2yrs: 0, emp_length: 5, home_ownership: 1,
};

const fieldLabels = {
  loan_amnt: 'Loan Amount ($)', int_rate: 'Interest Rate (%)', installment: 'Monthly Payment ($)',
  annual_inc: 'Annual Income ($)', dti: 'Debt-to-Income Ratio', fico_range_low: 'FICO Score (Low)',
  fico_range_high: 'FICO Score (High)', inq_last_6mths: 'Credit Inquiries (6mo)',
  open_acc: 'Open Accounts', pub_rec: 'Public Records', revol_bal: 'Revolving Balance ($)',
  revol_util: 'Revolving Utilization (%)', total_acc: 'Total Accounts',
  delinq_2yrs: 'Delinquencies (2yr)', emp_length: 'Employment Length (yrs)',
  home_ownership: 'Home Ownership (0=Rent, 1=Mortgage, 2=Own)',
};

export default function LoanForm() {
  const [form, setForm] = useState(defaultValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await predictLoan(form);
      setResult(res.data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>Loan Application</h2>
      <p style={styles.subtitle}>Submit an application to the trained federated model</p>

      <div style={styles.grid}>
        {Object.keys(defaultValues).map(key => (
          <div key={key} style={styles.field}>
            <label style={styles.label}>{fieldLabels[key]}</label>
            <input
              style={styles.input}
              type="number"
              value={form[key]}
              onChange={e => setForm({ ...form, [key]: parseFloat(e.target.value) })}
            />
          </div>
        ))}
      </div>

      <button onClick={handleSubmit} disabled={loading} style={styles.button}>
        {loading ? 'Evaluating...' : 'Submit Application'}
      </button>

      {result && (
        <div style={{ ...styles.result, borderColor: result.decision === 'APPROVED' ? '#0F6E56' : '#993C1D' }}>
          <div style={{ ...styles.decision, color: result.decision === 'APPROVED' ? '#0F6E56' : '#993C1D' }}>
            {result.decision === 'APPROVED' ? '✅ APPROVED' : '❌ DENIED'}
          </div>
          <div style={styles.probability}>
            Repayment probability: {(result.probability * 100).toFixed(1)}%
          </div>
          <div style={styles.reasons}>
            <div style={styles.reasonsTitle}>Reason Codes:</div>
            {result.reasons.map((r, i) => (
              <div key={i} style={styles.reason}>• {r}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: { background: '#12122a', borderRadius: 12, padding: 24, marginBottom: 24, border: '1px solid #2a2a3e' },
  title: { color: '#fff', fontSize: 18, fontWeight: 700, marginBottom: 4 },
  subtitle: { color: '#888', fontSize: 13, marginBottom: 20 },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 16, marginBottom: 24 },
  field: { display: 'flex', flexDirection: 'column', gap: 6 },
  label: { color: '#888', fontSize: 12 },
  input: { background: '#1a1a2e', border: '1px solid #2a2a3e', borderRadius: 6, padding: '8px 10px', color: '#fff', fontSize: 13 },
  button: { background: '#534AB7', color: '#fff', border: 'none', borderRadius: 8, padding: '12px 32px', fontSize: 14, fontWeight: 600, cursor: 'pointer' },
  result: { marginTop: 24, padding: 20, borderRadius: 10, border: '2px solid', background: '#0d0d1f' },
  decision: { fontSize: 24, fontWeight: 800, marginBottom: 8 },
  probability: { color: '#ccc', fontSize: 14, marginBottom: 12 },
  reasons: { display: 'flex', flexDirection: 'column', gap: 6 },
  reasonsTitle: { color: '#888', fontSize: 12, fontWeight: 600, marginBottom: 4 },
  reason: { color: '#ccc', fontSize: 13 },
};