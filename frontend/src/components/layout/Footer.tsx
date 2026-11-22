import { Link } from "react-router-dom";
import { timeAgo } from "../../lib/format";
import { useSiteStats } from "../../lib/useSiteStats";
import { MascotHead } from "../brand/Mascot";

const EXPLORE = [
  { to: "/", label: "Ask about rent" },
  { to: "/areas", label: "Area guides" },
  { to: "/compare", label: "Compare areas" },
  { to: "/move-in", label: "Move-in cost" },
  { to: "/saved", label: "Saved and favourites" },
];

export function Footer() {
  const stats = useSiteStats();
  const s = stats.data;
  return (
    <footer className="mt-16" style={{ background: "var(--forest)", color: "#e6efe9" }}>
      <div className="mx-auto grid max-w-[1200px] gap-10 px-4 py-12 sm:grid-cols-2 md:px-8 lg:grid-cols-[1.4fr_1fr_1fr_1.2fr]">
        <div className="space-y-3">
          <p className="flex items-center gap-3 font-display text-[22px] font-bold text-white"><MascotHead size={44} />Adugbo AI</p>
          <p className="max-w-[320px] text-[14px] leading-relaxed" style={{ color: "#c9d9d0" }}>
            Àdúgbò means neighbourhood. Adugbo AI tells you what rent in each Lagos neighbourhood really costs, worked out from listings posted in the last few weeks.
          </p>
          {s && s.listings > 0 && (
            <p className="tabular text-[13px] font-semibold text-white">{s.listings.toLocaleString("en-NG")} listings across {s.areas} areas</p>
          )}
        </div>
        <nav aria-label="Explore" className="space-y-3">
          <p className="text-[12px] font-bold uppercase tracking-[.08em]" style={{ color: "#9fc3b0" }}>Explore</p>
          <ul className="space-y-2 text-[14px]">
            {EXPLORE.map((l) => <li key={l.to}><Link to={l.to} className="text-white underline-offset-4 hover:underline">{l.label}</Link></li>)}
          </ul>
        </nav>
        <div className="space-y-3">
          <p className="text-[12px] font-bold uppercase tracking-[.08em]" style={{ color: "#9fc3b0" }}>How the numbers work</p>
          <ul className="space-y-2 text-[14px] leading-snug" style={{ color: "#dbe7e0" }}>
            <li>Every price comes from a real listing.</li>
            <li>Monthly prices are turned into yearly.</li>
            <li>Each answer says how many listings it uses.</li>
            <li>Too few listings? We say so.</li>
          </ul>
        </div>
        <div className="space-y-3">
          <p className="text-[12px] font-bold uppercase tracking-[.08em]" style={{ color: "#9fc3b0" }}>No account needed</p>
          <p className="text-[14px] leading-relaxed" style={{ color: "#dbe7e0" }}>
            There is no sign-up, on purpose. Your chats, saved searches and favourites stay linked to this browser only. We never ask for your name, email or phone number.
          </p>
        </div>
      </div>
      <div className="border-t" style={{ borderColor: "rgba(255,255,255,.12)" }}>
        <div className="mx-auto flex max-w-[1200px] flex-wrap items-center justify-between gap-3 px-4 py-5 text-[12px] md:px-8" style={{ color: "#b6cdbf" }}>
          <p>Prices from public listings on Nigerian property sites, refreshed weekly{s?.newest ? ` · newest listing ${timeAgo(s.newest)}` : ""}. Area notes are summarised from linked web sources.</p>
          <p>© {new Date().getFullYear()} Adugbo AI</p>
        </div>
      </div>
    </footer>
  );
}
