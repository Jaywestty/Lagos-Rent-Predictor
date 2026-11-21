import { useEffect, useMemo, useRef, useState } from "react";
import { nairaShort } from "../../lib/format";
import { track } from "../../services/api";

type Fees = { years: number; agencyPct: number; legalPct: number; cautionPct: number; service: number; other: number };

// Typical Lagos figures. They vary by landlord and agent, so every value is editable.
const DEFAULT_FEES: Fees = { years: 1, agencyPct: 10, legalPct: 10, cautionPct: 10, service: 0, other: 150_000 };

function NumberField({ label, value, onChange, suffix, step = 1, hint }: { label: string; value: number; onChange: (v: number) => void; suffix?: string; step?: number; hint?: string }) {
  return (
    <label className="block text-[13px] font-semibold">
      {label}
      <span className="mt-1 flex items-center gap-2">
        <input type="number" inputMode="numeric" min={0} step={step} className="input w-full tabular" value={Number.isFinite(value) ? value : 0}
          onChange={(e) => onChange(Math.max(0, Number(e.target.value) || 0))} />
        {suffix && <span className="muted text-[13px]">{suffix}</span>}
      </span>
      {hint && <span className="muted mt-1 block text-[12px] font-normal">{hint}</span>}
    </label>
  );
}

export function MoveInCalculator({ initialRent, area }: { initialRent?: number | null; area?: string | null }) {
  const [rent, setRent] = useState<number>(Math.round(initialRent ?? 1_500_000));
  const [fees, setFees] = useState<Fees>(DEFAULT_FEES);
  const tracked = useRef(false);

  useEffect(() => { if (initialRent) setRent(Math.round(initialRent)); }, [initialRent]);

  const parts = useMemo(() => {
    const rentDue = rent * fees.years;
    return [
      { key: "rent", label: `Rent (${fees.years} year${fees.years > 1 ? "s" : ""} upfront)`, value: rentDue, color: "var(--primary)" },
      { key: "agency", label: `Agency fee (${fees.agencyPct}%)`, value: (rent * fees.agencyPct) / 100, color: "var(--series-1)" },
      { key: "legal", label: `Legal / agreement fee (${fees.legalPct}%)`, value: (rent * fees.legalPct) / 100, color: "var(--series-2)" },
      { key: "caution", label: `Caution deposit (${fees.cautionPct}%, usually refundable)`, value: (rent * fees.cautionPct) / 100, color: "var(--series-3)" },
      { key: "service", label: "Service charge (estate, yearly)", value: fees.service, color: "var(--warn)" },
      { key: "other", label: "Moving and setup", value: fees.other, color: "var(--secondary)" },
    ].filter((p) => p.value > 0);
  }, [rent, fees]);
  const total = parts.reduce((s, p) => s + p.value, 0);

  function touched() {
    if (!tracked.current) { tracked.current = true; track("move_in_calculator", { area: area ?? null }); }
  }
  function update<K extends keyof Fees>(k: K, v: Fees[K]) {
    setFees((f) => ({ ...f, [k]: v }));
    touched();
  }

  return (
    <div className="grid gap-6 md:grid-cols-[1fr_1.1fr]">
      <div className="space-y-3">
        <NumberField label="Yearly rent" value={rent} onChange={(v) => { setRent(v); touched(); }} suffix="NGN" step={50_000} hint={initialRent ? `Filled in with the typical rent in ${area ?? "this area"}, from real listings.` : undefined} />
        <div>
          <p className="text-[13px] font-semibold">Years paid upfront</p>
          <div className="mt-1 flex gap-2">
            {[1, 2].map((y) => <button key={y} type="button" className="chip" aria-pressed={fees.years === y} onClick={() => update("years", y)}>{y} year{y > 1 ? "s" : ""}</button>)}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <NumberField label="Agency" value={fees.agencyPct} onChange={(v) => update("agencyPct", v)} suffix="%" />
          <NumberField label="Legal" value={fees.legalPct} onChange={(v) => update("legalPct", v)} suffix="%" />
          <NumberField label="Caution" value={fees.cautionPct} onChange={(v) => update("cautionPct", v)} suffix="%" />
        </div>
        <NumberField label="Service charge per year" value={fees.service} onChange={(v) => update("service", v)} suffix="NGN" step={50_000} />
        <NumberField label="Moving and setup" value={fees.other} onChange={(v) => update("other", v)} suffix="NGN" step={25_000} />
        <p className="muted text-[12px]">Percentages are common Lagos figures, not rules. Ask the agent for a written breakdown before you pay.</p>
      </div>
      <div className="card p-4 md:p-5" aria-live="polite">
        <p className="muted text-[13px]">Cash needed to move in</p>
        <p key={Math.round(total)} className="fade-in tabular text-[40px] font-bold leading-tight">{nairaShort(total)}</p>
        <p className="muted text-[13px] tabular">{total.toLocaleString("en-NG", { style: "currency", currency: "NGN", maximumFractionDigits: 0 })} · {rent ? (total / rent).toFixed(2) : "0"}× the yearly rent</p>
        <div className="mt-4 flex h-4 w-full gap-[2px] overflow-hidden rounded" role="img" aria-label="Cost breakdown">
          {parts.map((p) => <div key={p.key} className="grow-x h-4 first:rounded-l last:rounded-r" style={{ width: `${(p.value / total) * 100}%`, background: p.color }} title={`${p.label}: ${nairaShort(p.value)}`} />)}
        </div>
        <table className="mt-4 w-full text-[14px]">
          <tbody>
            {parts.map((p) => (
              <tr key={p.key} className="border-t" style={{ borderColor: "var(--border)" }}>
                <td className="py-2"><span className="mr-2 inline-block h-[10px] w-[10px] rounded-sm align-middle" style={{ background: p.color }} />{p.label}</td>
                <td className="tabular py-2 text-right font-semibold">{nairaShort(p.value)}</td>
              </tr>
            ))}
            <tr className="border-t" style={{ borderColor: "var(--border)" }}><td className="py-2 font-semibold">Total</td><td className="tabular py-2 text-right font-bold">{nairaShort(total)}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
