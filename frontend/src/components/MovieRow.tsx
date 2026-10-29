import React, { useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { MovieCard } from "./MovieCard";
import type { Movie, RecommendationItem } from "../api/types";

interface MovieRowProps {
  title: string;
  subtitle?: string;
  movies: (Movie | RecommendationItem)[];
  userRatingsMap?: Record<number, number>;
  watchlistMap?: Record<number, boolean>;
  onRatingSuccess?: () => void;
}

export const MovieRow: React.FC<MovieRowProps> = ({
  title,
  subtitle,
  movies,
  userRatingsMap = {},
  watchlistMap = {},
  onRatingSuccess,
}) => {
  const rowRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: -1 | 1) => {
    const row = rowRef.current;
    if (row) row.scrollBy({ left: direction * row.clientWidth * 0.75, behavior: "smooth" });
  };

  if (!movies || movies.length === 0) return null;

  return (
    <section className="space-y-3">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="font-display text-lg font-semibold text-white">{title}</h2>
          {subtitle && <p className="text-xs text-cinema-500">{subtitle}</p>}
        </div>
        <div className="hidden gap-1 sm:flex">
          <button
            onClick={() => scroll(-1)}
            className="rounded-md p-1.5 text-cinema-400 hover:bg-cinema-850 hover:text-white"
            aria-label="Scroll left"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            onClick={() => scroll(1)}
            className="rounded-md p-1.5 text-cinema-400 hover:bg-cinema-850 hover:text-white"
            aria-label="Scroll right"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div ref={rowRef} className="no-scrollbar flex gap-4 overflow-x-auto pb-2">
        {movies.map((m) => {
          // Rows accept both catalog movies and recommendation items.
          const id = "id" in m ? m.id : m.movie_id;

          return (
            <div key={id} className="w-36 flex-shrink-0 sm:w-44 md:w-48">
              <MovieCard
                id={id}
                title={m.title}
                year={m.year}
                genres={m.genres}
                poster_path={m.poster_path}
                userRating={userRatingsMap[id] || null}
                isInWatchlist={!!watchlistMap[id]}
                reason={"reason" in m ? m.reason : null}
                topSource={"top_source" in m ? m.top_source : null}
                onRatingSuccess={onRatingSuccess}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
};
