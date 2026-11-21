import { useState } from "react";
import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Ask", end: true },
  { to: "/areas", label: "Area guides" },
  { to: "/compare", label: "Compare" },
  { to: "/move-in", label: "Move-in cost" },
  { to: "/saved", label: "Saved" },
];

function linkClass({ isActive }: { isActive: boolean }) {
  return `relative min-h-[44px] content-center rounded-lg px-3 text-[14px] font-semibold transition-colors ${isActive ? "" : "muted hover:opacity-100"}`;
}

export function Header({ theme, onToggle }: { theme: string; onToggle: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-20 border-b backdrop-blur" style={{ background: "color-mix(in srgb, var(--surface) 88%, transparent)", borderColor: "var(--border)" }}>
      <div className="mx-auto flex h-[60px] max-w-[1080px] items-center justify-between gap-2 px-4 md:px-10">
        <NavLink to="/" className="flex items-center gap-2 text-[16px] font-bold">
          <span aria-hidden className="grid h-8 w-8 place-items-center rounded-lg text-[14px] text-white" style={{ background: "var(--forest)" }}>LR</span>
          <span className="hidden sm:inline">Lagos Rent</span>
        </NavLink>
        <nav aria-label="Main" className="hidden items-center md:flex">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={linkClass}>
              {({ isActive }) => (<>{l.label}{isActive && <span className="fade-in absolute inset-x-3 -bottom-[9px] h-[2px] rounded" style={{ background: "var(--primary)" }} />}</>)}
            </NavLink>
          ))}
        </nav>
        <div className="flex items-center gap-1">
          <a href="/admin" className="muted hidden min-h-[44px] content-center px-2 text-[13px] font-semibold md:block">Admin</a>
          <button type="button" onClick={onToggle} aria-label="Toggle theme" className="btn-ghost min-w-[44px] px-2">{theme === "light" ? "Dark" : "Light"}</button>
          <button type="button" className="btn-ghost md:hidden" aria-expanded={open} aria-controls="mobile-nav" onClick={() => setOpen((o) => !o)}>Menu</button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Main mobile" className="fade-in border-t px-4 pb-3 md:hidden" style={{ borderColor: "var(--border)" }}>
          {LINKS.map((l) => <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => `block ${linkClass({ isActive })}`} onClick={() => setOpen(false)}>{l.label}</NavLink>)}
          <a href="/admin" className="muted block min-h-[44px] content-center px-3 text-[14px] font-semibold">Admin</a>
        </nav>
      )}
    </header>
  );
}
