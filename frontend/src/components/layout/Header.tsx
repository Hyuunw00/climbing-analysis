export default function Header() {
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-border">
      <h1 className="text-sm font-bold tracking-widest uppercase">
        Climbing Analytics
      </h1>
      <div className="flex items-center gap-2">
        <span className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rec opacity-75" />
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-rec" />
        </span>
        <span className="text-xs font-mono text-rec">REC</span>
      </div>
    </header>
  );
}
