import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { AboutView } from './components/AboutView'
import { PanelLoader } from './components/AsyncPanels'
import { Banner } from './components/Banner'
import { LoadingScreen } from './components/LoadingScreen'
import { Navbar } from './components/Navbar'
import { MapView } from './components/MapView'
import { PerformanceView } from './components/PerformanceView'
import { ProfileView } from './components/ProfileView'
import { RegionalView } from './components/RegionalView'
import { Sidebar } from './components/Sidebar'
import type { Tab } from "./components/TabNav"
import { TimeSeriesView } from './components/TimeSeriesView'
import { connectDataSource, type DataSource } from './lib/dataSource'
import { useTheme } from './lib/theme'
import { defaultLocation, isOcean } from './lib/selectors'
import { MAP_VARIABLES, type MapVarKey } from './lib/types'

// three.js + @react-three/fiber is the bulk of the bundle — defer it until the user actually
// clicks a map cell, rather than paying for it on first load.
const DepthColumnModal = lazy(() => import('./components/DepthColumnModal').then((m) => ({ default: m.DepthColumnModal })))

export default function App() {
  const [src, setSrc] = useState<DataSource | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    connectDataSource().then(setSrc)
  }, [])

  // The loading screen holds until the data source has connected (real backend or mock fallback);
  // it also enforces its own minimum display time.
  if (!ready || !src) {
    return <LoadingScreen dataReady={src !== null} onComplete={() => setReady(true)} />
  }
  return <Dashboard src={src} />
}

function Dashboard({ src }: { src: DataSource }) {
  const ds = src.meta
  const initialLoc = useMemo(() => defaultLocation(ds), [ds])

  const [tab, setTab] = useState<Tab>('Maps')
  const [timeIdx, setTimeIdx] = useState(Math.floor(ds.times.length / 2))
  const [depthIdx, setDepthIdx] = useState(0)
  const [varKey, setVarKey] = useState<MapVarKey>('temp')
  const [lat, setLat] = useState(initialLoc.lat)
  const [lon, setLon] = useState(initialLoc.lon)
  const [controlsOpen, setControlsOpen] = useState(true)
  const [columnCell, setColumnCell] = useState<{ lat: number; lon: number } | null>(null)

  const varDef = MAP_VARIABLES.find((v) => v.key === varKey)!
  const onLand = !isOcean(ds, lat, lon)

  return (
    <div className="page-fade-in flex h-screen flex-col overflow-hidden">
      <Navbar
        active={tab}
        onChange={(t) => {
          setTab(t)
          // The Controls panel slides fully out on every tab but Maps (so the tab content gets the full
          // width) and comes back, expanded, when Maps is selected again.
          if (t === 'Maps') setControlsOpen(true)
        }}
      />

      <div className="flex min-h-0 w-full flex-1 flex-col gap-3 px-4 py-4 lg:px-6">
      <div className="shrink-0">
        <Banner warning={ds.attrs.syntheticWarning ?? undefined} />
      </div>

      <div className="scrollbar-thin flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto lg:flex-row lg:overflow-visible">
        <Sidebar
          ds={ds}
          timeIdx={timeIdx}
          onTimeIdx={setTimeIdx}
          depthIdx={depthIdx}
          onDepthIdx={setDepthIdx}
          varKey={varKey}
          onVarKey={setVarKey}
          lat={lat}
          lon={lon}
          onLat={setLat}
          onLon={setLon}
          onLand={onLand}
          visible={tab === 'Maps'}
          open={controlsOpen}
          onToggle={() => setControlsOpen((o) => !o)}
        />

        <main className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
          <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto pr-1">
            {tab === 'Maps' && (
              <MapView
                src={src}
                varDef={varDef}
                timeIdx={timeIdx}
                depthIdx={depthIdx}
                marker={{ lat, lon }}
                onCellClick={(clickLat, clickLon) => {
                  setLat(clickLat)
                  setLon(clickLon)
                  setColumnCell({ lat: clickLat, lon: clickLon })
                }}
              />
            )}
            {tab === 'Vertical profile' && <ProfileView src={src} timeIdx={timeIdx} lat={lat} lon={lon} onLand={onLand} />}
            {tab === 'Time series' && <TimeSeriesView src={src} depthIdx={depthIdx} lat={lat} lon={lon} onLand={onLand} />}
            {tab === 'Regional summary' && <RegionalView src={src} depthIdx={depthIdx} />}
            {tab === 'Model performance' && <PerformanceView />}
            {tab === 'About' && <AboutView />}
          </div>
        </main>
      </div>
      </div>

      {columnCell && (
        <Suspense fallback={<ModalLoadingFallback />}>
          <DepthColumnModal
            src={src}
            timeIdx={timeIdx}
            lat={columnCell.lat}
            lon={columnCell.lon}
            onClose={() => setColumnCell(null)}
          />
        </Suspense>
      )}
    </div>
  )
}

function ModalLoadingFallback() {
  const [theme] = useTheme()
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-abyss-800/40 backdrop-blur-sm" />
      <div className="relative flex items-center gap-3 rounded-xl border border-sand-400/70 bg-sand-300 px-5 py-3 text-sm font-medium text-abyss-800 shadow-xl dark:text-slate-100">
        <PanelLoader label="Loading 3D viewer" color={theme === 'dark' ? '#7fe3ea' : '#083a6b'} />
      </div>
    </div>
  )
}

