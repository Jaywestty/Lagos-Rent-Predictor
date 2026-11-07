export function Header({ theme, onToggle }: { theme: string; onToggle: () => void }) {
  return (
    <header className="sticky top-0 z-10 border-b" style={{ background: "var(--surface)", borderColor: "var(--border)", height: 60 }}>
      <div className="mx-auto flex h-full max-w-[760px] items-center justify-between px-4 md:px-10">
        <a href="/" className="text-[17px] font-semibold">Lagos Rent Assistant</a>
        <div className="flex items-center gap-2">
          <a href="/admin" className="min-h-[44px] content-center text-[14px] font-semibold px-2">Admin</a>
          <button type="button" onClick={onToggle} aria-label="Toggle theme" className="min-h-[44px] min-w-[44px] rounded border px-2 text-[14px] font-semibold" style={{ borderColor: "var(--border)" }}>
            {theme === "light" ? "Dark" : "Light"}
          </button>
        </div>
      </div>
    </header>
  );
}
