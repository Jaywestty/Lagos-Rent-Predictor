import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { timeAgo } from "../../lib/format";
import { useSiteStats } from "../../lib/useSiteStats";

const LINKS = [
  { to: "/", label: "Ask", end: true },
  { to: "/areas", label: "Area guides" },
  { to: "/compare", label: "Compare" },
  { to: "/move-in", label: "Move-in cost" },
  { to: "/saved", label: "Saved" },
];

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <span className="flex items-center gap-2.5">
      <span aria-hidden className="relative grid h-9 w-9 place-items-center overflow-hidden rounded-[10px] text-[13px] font-bold text-white" style={{ background: "var(--action)" }}>
        <span className="absolute inset-x-0 bottom-0 h-[38%]" style={{ background: "var(--forest)" }} />
        <span className="relative font-display">LR</span>
      </span>
      {!compact && (
        <span className="flex flex-col leading-none">
          <span className="font-display text-[17px] font-bold">Lagos Rent</span>
          <span className="muted mt-0.5 hidden text-[11px] font-medium sm:block">Real prices from real listings</span>
        </span>
      )}
    </span>
  );
}

function ThemeIcon({ dark }: { dark: boolean }) {
  return dark ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
  );
}

export function LiveBadge({ className = "" }: { className?: string }) {
  const stats = useSiteStats();
  if (!stats.data?.listings) return null;
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-[12px] font-semibold ${className}`} style={{ background: "var(--soft)", color: "var(--action)" }}>
      <span className="relative flex h-2 w-2" aria-hidden>
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: "var(--action)" }} />
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: "var(--action)" }} />
      </span>
      <span className="tabular">{stats.data.listings.toLocaleString("en-NG")} listings · updated {timeAgo(stats.data.newest)}</span>
    </span>
  );
}

function linkClass({ isActive }: { isActive: boolean }) {
  return `min-h-[40px] content-center rounded-lg px-3 text-[14px] font-semibold transition-colors ${isActive ? "" : "muted hover:text-[color:var(--text)]"}`;
}

export function Header({ theme, onToggle }: { theme: string; onToggle: () => void }) {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  const dark = theme === "dark";

  return (
    <header className="sticky top-0 z-20 border-b backdrop-blur" style={{ background: "color-mix(in srgb, var(--surface) 90%, transparent)", borderColor: "var(--border)" }}>
      <div className="mx-auto flex h-[64px] max-w-[1200px] items-center justify-between gap-3 px-4 md:px-8">
        <NavLink to="/" aria-label="Lagos Rent home"><Logo /></NavLink>
        <nav aria-label="Main" className="hidden items-center gap-1 lg:flex">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={linkClass} style={({ isActive }) => (isActive ? { background: "var(--soft)", color: "var(--action)" } : undefined)}>{l.label}</NavLink>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <LiveBadge className="hidden xl:inline-flex" />
          <button type="button" onClick={onToggle} aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} className="btn-ghost grid w-11 place-items-center px-0"><ThemeIcon dark={dark} /></button>
          <button type="button" className="btn-ghost grid w-11 place-items-center px-0 lg:hidden" aria-label="Menu" aria-expanded={open} aria-controls="mobile-nav" onClick={() => setOpen((o) => !o)}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>{open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h10" />}</svg>
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Main mobile" className="fade-in border-t px-4 pb-4 pt-2 lg:hidden" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <div className="grid gap-1">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => `block min-h-[48px] content-center rounded-lg px-3 text-[16px] font-semibold ${isActive ? "" : "muted"}`} style={({ isActive }) => (isActive ? { background: "var(--soft)", color: "var(--action)" } : undefined)}>{l.label}</NavLink>
            ))}
          </div>
          <LiveBadge className="mt-3" />
        </nav>
      )}
    </header>
  );
}
