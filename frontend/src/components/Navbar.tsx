import React, { useState } from "react";
import { NavLink, Link } from "react-router-dom";
import { Film, Menu, X } from "lucide-react";

const links = [
  { name: "Browse", path: "/" },
  { name: "For You", path: "/dashboard" },
  { name: "Search", path: "/search" },
  { name: "Ratings", path: "/ratings" },
  { name: "Watchlist", path: "/watchlist" },
  { name: "Benchmarks", path: "/analytics" },
];

export const Navbar: React.FC = () => {
  const [open, setOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-cinema-800 bg-cinema-950/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-crimson-500">
            <Film className="h-4 w-4 text-white" />
          </span>
          <span className="font-display text-lg font-bold tracking-tight text-white">
            CineMatch
          </span>
        </Link>

        <div className="hidden items-center gap-6 md:flex">
          {links.map((link) => (
            <NavLink
              key={link.path}
              to={link.path}
              end={link.path === "/"}
              className={({ isActive }) =>
                `relative py-4 text-sm transition-colors ${
                  isActive
                    ? "text-white after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:bg-crimson-500"
                    : "text-cinema-400 hover:text-white"
                }`
              }
            >
              {link.name}
            </NavLink>
          ))}
        </div>

        <Link
          to="/onboarding"
          className="hidden text-sm text-cinema-400 transition-colors hover:text-white md:block"
        >
          Edit taste profile
        </Link>

        <button
          onClick={() => setOpen(!open)}
          className="rounded-md p-2 text-cinema-400 hover:text-white md:hidden"
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {open && (
        <div className="border-t border-cinema-800 px-4 py-2 md:hidden">
          {[...links, { name: "Edit taste profile", path: "/onboarding" }].map((link) => (
            <NavLink
              key={link.path}
              to={link.path}
              end={link.path === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `block py-2.5 text-sm ${isActive ? "text-white" : "text-cinema-400"}`
              }
            >
              {link.name}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
};
