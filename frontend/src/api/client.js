// axios wrapper for all FastAPI endpoints
// all components import from here — one place to change the base URL

import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

export const getRounds = () => api.get('/training/rounds');
export const getStatus = () => api.get('/training/status');
export const getConvergence = () => api.get('/metrics/convergence');
export const getPrivacyTradeoff = () => api.get('/metrics/privacy');
export const getSummary = () => api.get('/metrics/summary');
export const predictLoan = (application) => api.post('/predict/', application);