import { createContext, useCallback, useContext, useState } from 'react'
import axios from 'axios'

const AuthContext = createContext(null)

function readStored() {
  try {
    const token = localStorage.getItem('cg_token')
    const user  = localStorage.getItem('cg_user')
    return { token: token || null, user: user ? JSON.parse(user) : null }
  } catch {
    return { token: null, user: null }
  }
}

export function AuthProvider({ children }) {
  const stored = readStored()
  const [token, setToken] = useState(stored.token)
  const [user,  setUser]  = useState(stored.user)

  const login = useCallback(async (email, password) => {
    const { data } = await axios.post('/api/auth/login', { email, password })
    const profile = {
      email:      data.email,
      role:       data.role,
      full_name:  data.full_name,
      patient_id: data.patient_id,
    }
    localStorage.setItem('cg_token', data.access_token)
    localStorage.setItem('cg_user',  JSON.stringify(profile))
    setToken(data.access_token)
    setUser(profile)
    return data
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('cg_token')
    localStorage.removeItem('cg_user')
    setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
