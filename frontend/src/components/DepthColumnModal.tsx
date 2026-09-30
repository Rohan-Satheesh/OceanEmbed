import { PanelLoader } from './AsyncPanels'
import { useTheme } from '../lib/theme'
import { DepthColumn3D } from './DepthColumn3D'
import { Modal } from './Modal'
import type { DataSource } from '../lib/dataSource'
import { useAsync } from '../lib/useAsync'

interface DepthColumnModalProps {
  src: DataSource
  timeIdx: number
  lat: number
  lon: number
  onClose: () => void
}

export function DepthColumnModal({ src, timeIdx, lat, lon, onClose }: DepthColumnModalProps) {
  const [theme] = useTheme()
  const { data: rows, error } = useAsync(() => src.profile(timeIdx, lat, lon), [src, timeIdx, lat, lon])

  return (
    <Modal title={`Depth profile — ${lat.toFixed(2)}N ${lon.toFixed(2)}E`} onClose={onClose}>
      {error && <p className="text-sm text-coral-600">Could not load this column: {error}</p>}
      {!rows && !error && (
        <div className="flex h-[420px] items-center justify-center text-abyss-800 dark:text-slate-100">
          <PanelLoader label="Loading column" color={theme === 'dark' ? '#7fe3ea' : '#083a6b'} />
        </div>
      )}
      {rows && <DepthColumn3D profile={rows} lat={lat} lon={lon} />}
    </Modal>
  )
}
