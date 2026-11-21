import { lazy, Suspense } from "react";
import { BrowserRouter, Outlet, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Header } from "./components/layout/Header";
import { Chat } from "./components/chat/Chat";
import { useTheme } from "./hooks/useTheme";
import { Skeleton } from "./components/states/States";

// Admin code is split out so public visitors never download it. Access needs an account made with the CLI.
const AdminLogin = lazy(() => import("./admin/AdminLogin").then((m) => ({ default: m.AdminLogin })));
const AdminLayout = lazy(() => import("./admin/AdminLayout").then((m) => ({ default: m.AdminLayout })));
const Analytics = lazy(() => import("./admin/Analytics").then((m) => ({ default: m.Analytics })));
const Operations = lazy(() => import("./admin/Operations").then((m) => ({ default: m.Operations })));
const Observability = lazy(() => import("./admin/Observability").then((m) => ({ default: m.Observability })));
const Areas = lazy(() => import("./pages/Areas").then((m) => ({ default: m.Areas })));
const AreaGuide = lazy(() => import("./pages/AreaGuide").then((m) => ({ default: m.AreaGuide })));
const Compare = lazy(() => import("./pages/Compare").then((m) => ({ default: m.Compare })));
const MoveIn = lazy(() => import("./pages/MoveIn").then((m) => ({ default: m.MoveIn })));
const Saved = lazy(() => import("./pages/Saved").then((m) => ({ default: m.Saved })));

const qc = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } });

function Page({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<div className="mx-auto max-w-[1080px] px-4 pt-8 md:px-10"><Skeleton rows={3} /></div>}>{children}</Suspense>;
}

function PublicLayout({ theme, onToggle }: { theme: string; onToggle: () => void }) {
  return (
    <>
      <Header theme={theme} onToggle={onToggle} />
      <main className="fade-in"><Outlet /></main>
      <footer className="muted mx-auto max-w-[1080px] px-4 pb-8 text-[12px] md:px-10">Rent figures come only from listings on NigeriaPropertyCentre and PropertyPro, refreshed weekly. Area notes are summarised from cited web sources.</footer>
    </>
  );
}

export default function App() {
  const { theme, toggle } = useTheme();
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route element={<PublicLayout theme={theme} onToggle={toggle} />}>
            <Route path="/" element={<Chat />} />
            <Route path="/areas" element={<Page><Areas /></Page>} />
            <Route path="/areas/:area" element={<Page><AreaGuide /></Page>} />
            <Route path="/compare" element={<Page><Compare /></Page>} />
            <Route path="/move-in" element={<Page><MoveIn /></Page>} />
            <Route path="/saved" element={<Page><Saved /></Page>} />
          </Route>
          <Route path="/admin/login" element={<Page><AdminLogin /></Page>} />
          <Route path="/admin" element={<Page><AdminLayout /></Page>}>
            <Route index element={<Page><Analytics /></Page>} />
            <Route path="observability" element={<Page><Observability /></Page>} />
            <Route path="operations" element={<Page><Operations /></Page>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
