import React from "react";
import { Link } from "react-router-dom";

export const Footer: React.FC = () => (
  <footer className="mt-20 border-t border-cinema-800 py-10 text-sm text-cinema-500">
    <div className="mx-auto grid max-w-7xl gap-8 px-4 sm:px-6 md:grid-cols-3 lg:px-8">
      <div className="space-y-2">
        <p className="font-display font-semibold text-white">CineMatch</p>
        <p className="max-w-xs leading-relaxed">
          Movie recommendations from your ratings, built on the MovieLens 100k dataset.
        </p>
      </div>

      <ul className="space-y-1.5">
        <li>
          <Link to="/" className="hover:text-white">
            Browse
          </Link>
        </li>
        <li>
          <Link to="/search" className="hover:text-white">
            Search
          </Link>
        </li>
        <li>
          <Link to="/analytics" className="hover:text-white">
            Model benchmarks
          </Link>
        </li>
      </ul>

      <p className="text-xs leading-relaxed">
        This product uses the TMDB API but is not endorsed or certified by TMDB. Posters and
        overviews courtesy of{" "}
        <a
          href="https://www.themoviedb.org"
          target="_blank"
          rel="noreferrer"
          className="text-cinema-400 underline-offset-2 hover:underline"
        >
          The Movie Database
        </a>
        .
      </p>
    </div>
  </footer>
);
