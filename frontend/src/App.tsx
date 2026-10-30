import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { AppStateProvider } from "./hooks/useAppState";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Analysis from "./pages/Analysis";
import Performance from "./pages/Performance";
import History from "./pages/History";

function NotFound() {
  return (
    <div className="panel p-10 text-center">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <Link to="/" className="btn-primary mt-6">Go to overview</Link>
    </div>
  );
}

export default function App() {
  return (
    <AppStateProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="upload" element={<Upload />} />
            <Route path="analysis" element={<Analysis />} />
            <Route path="performance" element={<Performance />} />
            <Route path="history" element={<History />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppStateProvider>
  );
}
