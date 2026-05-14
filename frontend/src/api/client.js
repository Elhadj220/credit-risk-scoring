import axios from 'axios'

const API_BASE = 'http://127.0.0.1:8000/api'

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' }
})

// Injecte automatiquement le token JWT dans chaque requête
client.interceptors.request.use(config => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export const login = async (username, password) => {
  const res = await client.post('/token/', { username, password })
  localStorage.setItem('access_token', res.data.access)
  localStorage.setItem('refresh_token', res.data.refresh)
  return res.data
}

export const predict = async (features) => {
  const res = await client.post('/predict/', features)
  return res.data
}

export const healthCheck = async () => {
  const res = await client.get('/health/')
  return res.data
}

export default client