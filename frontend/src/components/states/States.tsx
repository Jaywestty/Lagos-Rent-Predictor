export function LoadingState() {
  return <div className="card p-4 pulse" aria-label="Loading" role="status"><p className="text-[15px]">Checking listings...</p><div className="mt-3 h-4 rounded" style={{background:"var(--soft)"}} /><div className="mt-2 h-4 w-2/3 rounded" style={{background:"var(--soft)"}} /></div>;
}
export function NoDataState() {
  return <section className="card p-6" aria-label="No data"><h2 className="text-[17px] font-semibold">No listings found</h2><p className="mt-1 text-[15px]" style={{color:"var(--secondary)"}}>We have no data for this area yet. Try a nearby area or a different property type.</p></section>;
}
export function OutOfScopeState() {
  return <section className="card p-6" aria-label="Out of scope"><h2 className="text-[17px] font-semibold">Outside rent research</h2><p className="mt-1 text-[15px]" style={{color:"var(--secondary)"}}>This assistant covers annual rent in Lagos only.</p></section>;
}
export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <section className="card p-6" aria-label="Error"><h2 className="text-[17px] font-semibold">Something went wrong</h2><p className="mt-1 text-[15px]" style={{color:"var(--secondary)"}}>{message}</p><button type="button" onClick={onRetry} className="mt-3 min-h-[48px] rounded px-4 text-[14px] font-semibold text-white" style={{background:"var(--primary)"}}>Retry</button></section>;
}
