import { NavLink, Outlet } from "react-router-dom";
import { BarChart3, History, LayoutGrid, ScanSearch } from "lucide-react";
import Logo from "./Logo";
import ConnectionBadge from "./ConnectionBadge";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutGrid, end: true },
  { to: "/upload", label: "Analyze a video", icon: ScanSearch, end: false },
  { to: "/performance", label: "Model performance", icon: BarChart3, end: false },
  { to: "/history", label: "History", icon: History, end: false },
];

export default function Layout() {
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_1fr]">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-rule/70 bg-abyss/60 px-5 py-6 lg:flex">
        <Logo />
        <nav className="mt-10 flex flex-col gap-1" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `group flex items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] transition ${
                  isActive ? "bg-deck text-ink" : "text-haze hover:bg-deck/60 hover:text-ink"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon size={18} className={isActive ? "text-signal" : "text-haze group-hover:text-ink"} />
                  {label}
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto">
          <ConnectionBadge detailed />
        </div>
      </aside>

      {/* Mobile header */}
      <header className="sticky top-0 z-20 border-b border-rule/70 bg-abyss/90 backdrop-blur lg:hidden">
        <div className="flex items-center justify-between px-4 py-3">
          <Logo compact />
          <ConnectionBadge />
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-2" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex shrink-0 items-center gap-2 rounded-lg px-3 py-1.5 text-sm ${
                  isActive ? "bg-deck text-ink" : "text-haze"
                }`
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-10 lg:py-10">
        <Outlet />
      </main>
    </div>
  );
}
