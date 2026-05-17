const METRICS = ["Route Color", "Vertical Velocity", "Smoothness", "Technique", "Stable", "Adhesion"];

export default function DashboardPanel() {
  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border bg-panel p-4 lg:w-80">
      <h2 className="text-sm font-bold tracking-widest uppercase text-accent">
        Climbing Analytics
      </h2>
      <div className="flex flex-col gap-3">
        {METRICS.map((metric) => (
          <div key={metric} className="flex flex-col gap-1">
            <span className="text-xs font-mono uppercase text-muted">{metric}</span>
            <div className="h-2 rounded-full bg-border" />
          </div>
        ))}
      </div>
      <div className="mt-2 border-t border-border pt-4">
        <span className="text-xs font-mono uppercase text-muted">Live Score</span>
        <p className="text-4xl font-bold text-accent">--<span className="text-lg text-muted"> / 100</span></p>
      </div>
    </div>
  );
}
