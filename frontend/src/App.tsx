import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Header } from "./components/layout/Header";
import { Chat } from "./components/chat/Chat";
import { useTheme } from "./hooks/useTheme";
import { AdminLogin } from "./admin/AdminLogin";
import { AdminLayout } from "./admin/AdminLayout";

const Analytics = lazy(() => import("./admin/Analytics").then((m) => ({ default: m.Analytics })));
const Operations = lazy(() => import("./admin/Operations").then((m) => ({ default: m.Operations })));

const qc = new QueryClient();

export default function App() {
  const { theme, toggle } = useTheme();
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<><Header theme={theme} onToggle={toggle} /><main><Chat /></main></>} />
          <Route path="/admin/login" element={<AdminLogin />} />
          <Route path="/admin" element={<AdminLayout />}>
            <Route index element={<Suspense fallback={<p className="pulse">Loading analytics...</p>}><Analytics /></Suspense>} />
            <Route path="operations" element={<Suspense fallback={<p className="pulse">Loading operations...</p>}><Operations /></Suspense>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
