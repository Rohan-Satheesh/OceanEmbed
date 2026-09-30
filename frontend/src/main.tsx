import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { initTheme } from './lib/theme'
import { WaveBackground } from './components/background/WaveBackground'

initTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WaveBackground />
    <App />
  </StrictMode>,
)
