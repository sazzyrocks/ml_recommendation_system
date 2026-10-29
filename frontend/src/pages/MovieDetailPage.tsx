import React, { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Bookmark, Star, ChevronLeft, Check, Link2 } from "lucide-react";
import { getMovie, getSimilarMovies } from "../api/movies";
import { createRating, updateRating } from "../api/ratings";
import { addToWatchlist, removeFromWatchlist } from "../api/watchlist";
import { StarRating } from "../components/StarRating";
import { MoviePoster } from "../components/MoviePoster";
import { MovieCard } from "../components/MovieCard";
import { CardSkeleton, EmptyState } from "../components/LoadingSkeleton";
import { useAuth } from "../context/AuthContext";
import { useLibrary } from "../hooks/useLibrary";

export const MovieDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const movieId = parseInt(id || "0", 10);
  const { isAuthenticated } = useAuth();
  const { ratingsById, watchlistIds } = useLibrary();
  const queryClient = useQueryClient();
  const [copied, setCopied] = useState(false);

  const {
    data: movie,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["movie", movieId],
    queryFn: () => getMovie(movieId),
    enabled: movieId > 0,
  });

  const { data: similar, isLoading: similarLoading } = useQuery({
    queryKey: ["movie", movieId, "similar"],
    queryFn: () => getSimilarMovies(movieId, 10),
    enabled: movieId > 0,
  });

  const existingRating = ratingsById[movieId] ?? null;
  const isWatchlisted = !!watchlistIds[movieId];

  const watchlistMutation = useMutation({
    mutationFn: async () => {
      if (isWatchlisted) await removeFromWatchlist(movieId);
      else await addToWatchlist(movieId);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["watchlist"] }),
  });

  const ratingMutation = useMutation({
    mutationFn: (stars: number) =>
      existingRating !== null ? updateRating(movieId, stars) : createRating(movieId, stars),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ratings"] });
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });

  const requireAuth = (action: () => void) => {
    if (!isAuthenticated) {
      window.location.href = "/login";
      return;
    }
    action();
  };

  const copyLink = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl animate-pulse px-4 py-10 sm:px-6 lg:px-8">
        <div className="flex gap-8">
          <div className="aspect-[2/3] w-56 rounded-lg bg-cinema-850" />
          <div className="flex-1 space-y-3">
            <div className="h-8 w-1/2 rounded bg-cinema-850" />
            <div className="h-4 w-1/4 rounded bg-cinema-850" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !movie) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-16">
        <EmptyState
          title="Movie not found"
          description="This title isn't in the catalog, or the API couldn't be reached."
          actionText="Back to browse"
          actionHref="/"
        />
      </div>
    );
  }

  const backdropUrl = movie.poster_path
    ? `https://image.tmdb.org/t/p/w780${movie.poster_path}`
    : null;

  return (
    <div className="mx-auto max-w-7xl space-y-14 px-4 py-8 sm:px-6 lg:px-8">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-cinema-400 hover:text-white"
      >
        <ChevronLeft className="h-4 w-4" />
        Browse
      </Link>

      <div className="relative isolate overflow-hidden rounded-xl border border-cinema-800 p-6 sm:p-8">
        {backdropUrl && (
          <div
            className="absolute inset-0 -z-10 scale-110 bg-cover bg-center opacity-10 blur-2xl"
            style={{ backgroundImage: `url(${backdropUrl})` }}
          />
        )}

        <div className="flex flex-col gap-8 md:flex-row">
          <div className="aspect-[2/3] w-48 shrink-0 overflow-hidden rounded-lg ring-1 ring-cinema-700 sm:w-60">
            <MoviePoster
              title={movie.title}
              year={movie.year}
              genres={movie.genres}
              poster_path={movie.poster_path}
              size="lg"
            />
          </div>

          <div className="flex-1 space-y-5">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl">
                {movie.title}
              </h1>
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-cinema-400">
                {movie.year && <span>{movie.year}</span>}
                {movie.genres.length > 0 && <span>{movie.genres.join(", ")}</span>}
                <span className="flex items-center gap-1 text-gold-400">
                  <Star className="h-3.5 w-3.5 fill-gold-400" />
                  <span className="tabular-nums">
                    {movie.average_rating ? movie.average_rating.toFixed(1) : "—"}
                  </span>
                  <span className="text-cinema-500">
                    ({movie.ratings_count.toLocaleString()} ratings)
                  </span>
                </span>
              </div>
            </div>

            <p className="max-w-3xl leading-relaxed text-cinema-200">
              {movie.overview || "No overview available."}
            </p>

            <div className="flex flex-wrap items-center gap-3 border-t border-cinema-800 pt-5">
              <div className="flex items-center gap-2 text-sm text-cinema-400">
                <span>{existingRating ? "Your rating" : "Rate it"}</span>
                <StarRating
                  rating={existingRating || 0}
                  readOnly={false}
                  onChange={(stars) => requireAuth(() => ratingMutation.mutate(stars))}
                  showNumeric
                />
              </div>

              <button
                onClick={() => requireAuth(() => watchlistMutation.mutate())}
                disabled={watchlistMutation.isPending}
                className={
                  isWatchlisted
                    ? "btn bg-emerald-600 text-white hover:bg-emerald-600/90"
                    : "btn-secondary"
                }
              >
                <Bookmark className={`h-4 w-4 ${isWatchlisted ? "fill-white" : ""}`} />
                {isWatchlisted ? "On your watchlist" : "Add to watchlist"}
              </button>

              <button onClick={copyLink} className="btn-secondary px-2.5" aria-label="Copy link">
                {copied ? (
                  <Check className="h-4 w-4 text-emerald-400" />
                ) : (
                  <Link2 className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      <section className="space-y-4">
        <div>
          <h2 className="font-display text-xl font-semibold text-white">More like this</h2>
          <p className="text-xs text-cinema-500">Similar genres, tags and release era.</p>
        </div>

        {similarLoading ? (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <CardSkeleton key={i} />
            ))}
          </div>
        ) : similar && similar.length > 0 ? (
          <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {similar.map((sim) => (
              <MovieCard
                key={sim.id}
                id={sim.id}
                title={sim.title}
                year={sim.year}
                genres={sim.genres}
                poster_path={sim.poster_path}
                similarityScore={sim.similarity}
                userRating={ratingsById[sim.id] || null}
                isInWatchlist={!!watchlistIds[sim.id]}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-cinema-500">No similar titles found.</p>
        )}
      </section>
    </div>
  );
};
