import { ErrorPanel, LoadingPanel } from './AsyncPanels'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DataSource } from '../lib/dataSource'
import type { RegionalPoint } from '../lib/selectors'
import { REGIONS } from '../lib/types'
import { useAsync } from '../lib/useAsync'

interface RegionalViewProps {
  src: DataSource
  depthIdx: number
}

export function RegionalView({ src, depthIdx }: RegionalViewProps) {
  const ds = src.meta
  const { data, error } = useAsync(async () => {
    const keys = Object.keys(REGIONS) as (keyof typeof REGIONS)[]
    const results = await Promise.all(keys.map((k) => src.regional(k, depthIdx)))
    return Object.fromEntries(keys.map((k, i) => [k, results[i]])) as Record<string, RegionalPoint[]>
  }, [src, depthIdx])

  if (error) return <ErrorPanel message={error} />
  if (!data) return <LoadingPanel />

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      {(Object.keys(REGIONS) as (keyof typeof REGIONS)[]).map((region) => {
        const rows = data[region as string]
        const valid = rows.filter((r) => Number.isFinite(r.rmse))
        const meanRmse = valid.length ? valid.reduce((s, r) => s + r.rmse, 0) / valid.length : NaN

        return (
          <div key={region} className="glass-panel rounded-2xl p-5">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-100">{REGIONS[region].label}</h3>
              <div className="rounded-lg border border-abyss-600 bg-abyss-900 px-3 py-1.5 text-right">
                <div className="text-[10px] uppercase tracking-wide text-slate-300">Mean daily RMSE vs GLORYS</div>
                <div className="text-sm font-semibold text-surf-400">{Number.isFinite(meanRmse) ? `${meanRmse.toFixed(3)} deg C` : 'n/a'}</div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={rows} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid stroke="#1b4657" strokeDasharray="3 3" />
                <XAxis dataKey="time" stroke="#e2e8f0" fontSize={10} />
                <YAxis
                  stroke="#e2e8f0"
                  fontSize={11}
                  domain={['auto', 'auto']}
                  label={{ value: `regional mean (deg C) @ ${ds.depths[depthIdx]}m`, angle: -90, position: 'insideLeft', fill: '#cbd5e1', fontSize: 10 }}
                />
                <Tooltip contentStyle={{ background: '#0c2836', border: '1px solid #1b4657', borderRadius: 10, fontSize: 12 }} labelStyle={{ color: '#e2e8f0' }} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#e2e8f0' }} />
                <Line type="monotone" dataKey="reconstruction" stroke="#38e3d8" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="glorys" stroke="#ff7a59" strokeWidth={2} strokeDasharray="5 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      })}
    </div>
  )
}
