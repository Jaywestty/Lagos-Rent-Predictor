import { useEffect, useState } from "react";
import { Link, Outlet, useNavigate } from "react-router-dom";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

export function AdminLayout() {
  const [ok, setOk] = useState(false);
  const navigate = useNavigate();
  useEffect(() => {
    fetch(`${BASE}/admin/auth/me`, { credentials: "include" }).then((r) => {
      if (!r.ok) navigate("/admin/login", { replace: true });
      else setOk(true);
    }).catch(() => navigate("/admin/login", { replace: true }));
  }, [navigate]);
  if (!ok) return <main className="p-6"><p className="pulse">Checking sign in...</p></main>;
  return (
    <div className="flex min-h-screen">
      <aside className="w-[240px] shrink-0 border-r p-4 max-md:w-[72px]" style={{ background: "var(--surface)", borderColor: "var(--border)" }} aria-label="Admin navigation">
        <p className="text-[17px] font-semibold">Admin</p>
        <nav className="mt-4 grid gap-2">
          <Link className="min-h-[44px] content-center rounded px-2 text-[14px] font-semibold" to="/admin">Analytics</Link>
          <Link className="min-h-[44px] content-center rounded px-2 text-[14px] font-semibold" to="/admin/operations">Operations</Link>
        </nav>
      </aside>
      <main className="min-w-0 flex-1 p-4 md:p-8">
        <Outlet />
      </main>
    </div>
  );
}
