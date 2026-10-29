import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Bookmark, Star } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { addToWatchlist, removeFromWatchlist } from "../api/watchlist";
import { createRating, updateRating } from "../api/ratings";
import { useAuth } from "../context/AuthContext";
import { StarRating } from "./StarRating";
import { MoviePoster } from "./MoviePoster";

interface MovieCardProps {
  id: number;
  title: string;
  year?: number | null;
  genres?: string[];
  poster_path?: string | null;
  userRating?: number | null;
  isInWatchlist?: boolean;
  reason?: string | null;
  topSource?: string | null;
  similarityScore?: number | null;
  onRatingSuccess?: () => void;
}

export const MovieCard: React.FC<MovieCardProps> = ({
  id,
  title,
  year,
  genres = [],
  poster_path,
  userRating = null,
  isInWatchlist = false,
  reason = null,
  topSource = null,
  similarityScore = null,
  onRatingSuccess,
}) => {
  const { isAuthenticated } = useAuth();
  const queryClient = useQueryClient();
  const [showRating, setShowRating] = useState(false);
  const [isWatchlisted, setIsWatchlisted] = useState(isInWatchlist);
  const [currentRating, setCurrentRating] = useState<number | null>(userRating);

  const watchlistMutation = useMutation({
    mutationFn: async () => {
      if (isWatchlisted) {
        await removeFromWatchlist(id);
        return false;
      }
      await addToWatchlist(id);
      return true;
    },
    onSuccess: (saved) => {
      setIsWatchlisted(saved);
      queryClient.invalidateQueries({ queryKey: ["watchlist"] });
    },
  });

  const rateMutation = useMutation({
    mutationFn: (stars: number) =>
      currentRating !== null ? updateRating(id, stars) : createRating(id, stars),
    onSuccess: (data) => {
      setCurrentRating(data.rating);
      setShowRating(false);
      // A new rating changes the user's fold-in vector, so recommendations are stale.
      queryClient.invalidateQueries({ queryKey: ["ratings"] });
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      onRatingSuccess?.();
    },
  });

  const handleRate = (stars: number) => {
    if (!isAuthenticated) {
      window.location.href = "/login";
      return;
    }
    rateMutation.mutate(stars);
  };

  const handleWatchlistClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isAuthenticated) {
      window.location.href = "/login";
      return;
    }
    watchlistMutation.mutate();
  };

  return (
    <div className="group relative flex w-full flex-col">
      <Link
        to={`/movies/${id}`}
        className="relative block aspect-[2/3] w-full overflow-hidden rounded-lg bg-cinema-850 ring-1 ring-cinema-800 transition group-hover:ring-cinema-600"
      >
        <MoviePoster title={title} year={year} genres={genres} poster_path={poster_path} />

        {similarityScore !== null && (
          <span className="absolute left-2 top-2 rounded bg-cinema-950/85 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-accent-cyan">
            {(similarityScore * 100).toFixed(0)}% match
          </span>
        )}

        <button
          onClick={handleWatchlistClick}
          disabled={watchlistMutation.isPending}
          className={`absolute right-2 top-2 rounded-md p-1.5 transition ${
            isWatchlisted
              ? "bg-crimson-500 text-white"
              : "bg-cinema-950/80 text-cinema-200 opacity-0 hover:text-white group-hover:opacity-100 focus:opacity-100"
          }`}
          aria-label={isWatchlisted ? "Remove from watchlist" : "Add to watchlist"}
        >
          <Bookmark className={`h-3.5 w-3.5 ${isWatchlisted ? "fill-white" : ""}`} />
        </button>
      </Link>

      <div className="mt-2.5 flex flex-1 flex-col">
        <Link
          to={`/movies/${id}`}
          className="line-clamp-1 text-sm font-medium text-white hover:text-crimson-400"
          title={title}
        >
          {title}
        </Link>

        <div className="mt-0.5 flex items-center justify-between text-xs text-cinema-500">
          <span className="truncate">{[year, genres[0]].filter(Boolean).join(" · ")}</span>

          <div className="relative shrink-0">
            <button
              onClick={() => setShowRating(!showRating)}
              className={`flex items-center gap-1 tabular-nums transition-colors ${
                currentRating ? "text-gold-400" : "hover:text-gold-400"
              }`}
            >
              <Star className={`h-3 w-3 ${currentRating ? "fill-gold-400" : ""}`} />
              {currentRating ? currentRating.toFixed(1) : "Rate"}
            </button>

            {showRating && (
              <div className="absolute bottom-full right-0 z-30 mb-2 rounded-lg border border-cinema-700 bg-cinema-900 p-2 shadow-xl">
                <StarRating
                  rating={currentRating || 0}
                  readOnly={false}
                  size="sm"
                  onChange={handleRate}
                />
              </div>
            )}
          </div>
        </div>

        {reason && (
          <p className="mt-2 line-clamp-2 text-xs leading-snug text-cinema-400" title={reason}>
            {reason}
            {topSource && (
              <span className="text-cinema-600"> · {topSource.replaceAll("_", " ")}</span>
            )}
          </p>
        )}
      </div>
    </div>
  );
};
