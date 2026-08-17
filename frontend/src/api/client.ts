import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''  // relative for proxy

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
})

export type DatasetId = string

export interface Sample {
  label: string
  filename: string
  rows_hint?: string
}

export interface UploadResponse {
  dataset_id: string
  name: string
  rows: number
  cols: number
  notes: string[]
  sheets?: string[]
  source: string
}

export interface ProfileResponse {
  dataset_id: string
  roles: Record<string, string>
  summary: any
  column_profile: any[]
  structure_hint: string
  prep_notes: string[]
  numeric_describe: any[]
}

// Data
export async function listSamples(): Promise<Sample[]> {
  const res = await api.get('/api/v1/data/samples')
  return res.data.samples
}
export async function uploadFile(file: File): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await api.post('/api/v1/data/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return res.data
}
export async function loadUrl(url: string): Promise<UploadResponse> {
  const res = await api.post('/api/v1/data/url', { url })
  return res.data
}
export async function loadSample(label: string): Promise<UploadResponse> {
  const res = await api.post(`/api/v1/data/sample/${encodeURIComponent(label)}`)
  return res.data
}
export async function prepareDataset(dataset_id: string, sheet?: string): Promise<ProfileResponse> {
  const res = await api.post('/api/v1/data/prepare', { dataset_id, sheet })
  return res.data
}
export async function overrideRoles(dataset_id: string, roles: Record<string,string>): Promise<ProfileResponse> {
  const res = await api.post('/api/v1/profile/override', { dataset_id, roles })
  return res.data
}

// Analysis
export async function getKpi(dataset_id: string, metric?: string, date_col?: string, filters?: any) {
  const res = await api.post('/api/v1/analysis/kpi', { dataset_id, metric, date_col, filters })
  return res.data
}
export async function getCorrelation(dataset_id: string, method='pearson', filters?: any) {
  const res = await api.post('/api/v1/analysis/correlation', { dataset_id, method, filters })
  return res.data
}
export async function getGroupStats(dataset_id: string, metric: string, group_col: string, filters?: any) {
  const res = await api.post('/api/v1/analysis/group-stats', { dataset_id, metric, group_col, filters })
  return res.data
}
export async function getAnova(dataset_id: string, metric: string, group_col: string, filters?: any) {
  const res = await api.post('/api/v1/analysis/anova', { dataset_id, metric, group_col, filters })
  return res.data
}
export async function getChiSquare(dataset_id: string, col_a: string, col_b: string, filters?: any) {
  const res = await api.post('/api/v1/analysis/chi-square', { dataset_id, col_a, col_b, filters })
  return res.data
}
export async function getOutliers(dataset_id: string, filters?: any) {
  const res = await api.post('/api/v1/analysis/outliers', { dataset_id, filters })
  return res.data
}
export async function getTrend(dataset_id: string, date_col: string, value_col: string, group_col?: string, agg='sum', freq='M', filters?: any) {
  const res = await api.post('/api/v1/analysis/trend', { dataset_id, date_col, value_col, group_col, agg, freq, filters })
  return res.data
}

// Modeling
export async function runRegression(dataset_id: string, target: string, features: string[], filters?: any) {
  const res = await api.post('/api/v1/modeling/regression', { dataset_id, target, features, filters })
  return res.data
}
export async function runClassification(dataset_id: string, target: string, features: string[], filters?: any) {
  const res = await api.post('/api/v1/modeling/classification', { dataset_id, target, features, filters })
  return res.data
}
export async function runClustering(dataset_id: string, features: string[], k_min=2, k_max=8, filters?: any) {
  const res = await api.post('/api/v1/modeling/clustering', { dataset_id, features, k_min, k_max, filters })
  return res.data
}
export async function runForecast(dataset_id: string, date_col: string, value_col: string, periods=12, freq='M', agg='sum', filters?: any) {
  const res = await api.post('/api/v1/modeling/forecast', { dataset_id, date_col, value_col, periods, freq, agg, filters })
  return res.data
}

// Plots
export async function generatePlot(dataset_id: string, chart_type: string, params: any, filters?: any) {
  const res = await api.post('/api/v1/plots/generate', { dataset_id, chart_type, params, filters })
  return res.data
}

export async function getDictionary(dataset_id: string) {
  const res = await api.get(`/api/v1/profile/dictionary/${dataset_id}`)
  return res.data
}
