import { useState } from "react";
import { useNavigate } from "react-router-dom";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

export function AdminLogin() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch(`${BASE}/admin/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }), credentials: "include" });
    if (!res.ok) { setError("Invalid email or password"); return; }
    const body = await res.json();
    localStorage.setItem("admin-csrf", body.csrf_token || "");
    navigate("/admin");
  }
  return (
    <main className="mx-auto max-w-[760px] p-4 md:p-10">
      <form className="card p-6" onSubmit={submit} aria-label="Admin login">
        <h1 className="text-[22px] font-semibold">Admin login</h1>
        {error && <p role="alert" className="mt-2 text-[15px]">{error}</p>}
        <label className="mt-4 block text-[12px] font-semibold" htmlFor="email">Email</label>
        <input id="email" className="mt-1 h-[52px] w-full rounded border px-3 text-[15px]" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="username" />
        <label className="mt-3 block text-[12px] font-semibold" htmlFor="password">Password</label>
        <input id="password" type="password" className="mt-1 h-[52px] w-full rounded border px-3 text-[15px]" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
        <button type="submit" className="mt-4 min-h-[48px] w-full rounded text-[14px] font-semibold text-white" style={{ background: "var(--primary)" }}>Sign in</button>
      </form>
    </main>
  );
}
