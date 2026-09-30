export function AboutView() {
  return (
    <div className="glass-panel space-y-5 rounded-2xl p-6 text-sm leading-relaxed text-slate-300">
      <Section title="What this is">
        OceanEmbed learns an embedding of six daily surface satellite variables (SST, SSS, SSH, currents, winds) and
        decodes it to temperature at 15 depths.
      </Section>
      <Section title="How to read it">
        The uncertainty layer is a Monte-Carlo-dropout spread; check the reliability plot in{' '}
        <em>Model performance</em> before treating it as a calibrated interval. MLD, D20 and TCHP are coarse
        estimates from 15 vertical levels.
      </Section>
      <Section title="Limits">
        Scores are against GLORYS only. A model trained on few years, or on synthetic data, demonstrates the
        pipeline rather than real-ocean skill.
      </Section>
      <div className="rounded-xl border border-abyss-600 bg-abyss-900/60 p-4 text-xs text-slate-300">
        This is the React frontend scaffold (in development, built against synthetic mock data pending Chunks 4-6).
        The Streamlit dashboard at <code className="text-surf-400">dashboard/app.py</code> remains the working
        reference implementation until migration is explicitly requested.
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-surf-400">{title}</h3>
      <p>{children}</p>
    </div>
  )
}
