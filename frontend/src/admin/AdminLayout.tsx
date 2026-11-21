import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";

const BASE = import.meta.env.VITE_API_BASE_URL || "";
const LINKS = [
  { to: "/admin", label: "Analytics", short: "Stats", end: true },
  { to: "/admin/observability", label: "Observability", short: "Health" },
  { to: "/admin/operations", label: "Operations", short: "Ops" },
];

export function AdminLayout() {
  const [me, setMe] = useState<{ email: string } | null>(null);
  const navigate = useNavigate();
  useEffect(() => {
    fetch(`${BASE}/api/admin/auth/me`, { credentials: "include" }).then(async (r) => {
      if (!r.ok) navigate("/admin/login", { replace: true });
      else setMe(await r.json());
    }).catch(() => navigate("/admin/login", { replace: true }));
  }, [navigate]);

  async function logout() {
    await fetch(`${BASE}/api/admin/auth/logout`, { method: "POST", credentials: "include", headers: { "X-CSRF-Token": localStorage.getItem("admin-csrf") || "" } }).catch(() => undefined);
    localStorage.removeItem("admin-csrf");
    navigate("/admin/login", { replace: true });
  }

  if (!me) return <main className="p-6"><div className="skeleton h-8 w-48" /><div className="skeleton mt-4 h-32" /></main>;
  return (
    <div className="flex min-h-screen">
      <aside className="flex w-[220px] shrink-0 flex-col border-r p-4 max-md:w-[84px] max-md:p-2" style={{ background: "var(--surface)", borderColor: "var(--border)" }} aria-label="Admin navigation">
        <p className="text-[16px] font-bold max-md:text-center max-md:text-[13px]">Admin</p>
        <nav className="mt-4 grid gap-1">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => `min-h-[44px] content-center rounded-lg px-3 text-[14px] font-semibold transition-colors max-md:px-1 max-md:text-center max-md:text-[12px] ${isActive ? "" : "muted"}`} style={({ isActive }) => (isActive ? { background: "var(--soft)" } : undefined)}>
              <span className="max-md:hidden">{l.label}</span><span className="md:hidden">{l.short}</span>
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto space-y-1 pt-4 text-[12px]">
          <p className="muted truncate max-md:hidden" title={me.email}>{me.email}</p>
          <Link to="/" className="muted block min-h-[36px] content-center hover:underline">← Site</Link>
          <button type="button" onClick={logout} className="min-h-[36px] font-semibold" style={{ color: "var(--danger)" }}>Sign out</button>
        </div>
      </aside>
      <main className="fade-in min-w-0 flex-1 p-4 md:p-8">
        <Outlet />
      </main>
    </div>
  );
}
