import { useEffect, useState, type DependencyList } from 'react'

export interface AsyncState<T> {
  data: T | undefined
  loading: boolean
  error: string | null
}

/** Runs `fn` whenever `deps` change; stale responses (from a superseded call) are dropped. While a new
 * request is in flight the previous `data` is kept, so charts and maps don't blank on every slider tick. */
export function useAsync<T>(fn: () => Promise<T>, deps: DependencyList): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ data: undefined, loading: true, error: null })

  useEffect(() => {
    let cancelled = false
    setState((s) => ({ ...s, loading: true, error: null }))
    fn().then(
      (data) => {
        if (!cancelled) setState({ data, loading: false, error: null })
      },
      (err) => {
        if (!cancelled) setState((s) => ({ ...s, loading: false, error: err instanceof Error ? err.message : String(err) }))
      },
    )
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}
