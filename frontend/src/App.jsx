import { useState } from 'react'
import LoginForm from './components/LoginForm'
import DashboardPage from './pages/DashboardPage'

export default function App() {
  const [isAuth, setIsAuth] = useState(
    !!localStorage.getItem('access_token')
  )

  const handleLogin = () => setIsAuth(true)

  const handleLogout = () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    setIsAuth(false)
  }

  return isAuth
    ? <DashboardPage onLogout={handleLogout} />
    : <LoginForm onSuccess={handleLogin} />
}