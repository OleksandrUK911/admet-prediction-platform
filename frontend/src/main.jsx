import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './theme.css'
import { applyStoredTheme } from './theme'
import './i18n'
import App from './App.jsx'

// Apply any explicitly-chosen theme before the first paint to avoid a
// flash of the wrong theme. With no stored preference, theme.css falls
// back to following the OS `prefers-color-scheme`.
applyStoredTheme()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
