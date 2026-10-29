import React from "react";
import { CardSkeleton, EmptyState } from "../components/LoadingSkeleton";
import { MovieCard } from "../components/MovieCard";
import { PageHeader } from "../components/PageHeader";
import { useLibrary } from "../hooks/useLibrary";

const GRID = "grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5";

export const WatchlistPage: React.FC = () => {
  const { watchlist, ratingsById } = useLibrary();
  const entries = watchlist.data ?? [];

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6 lg:px-8">
      <PageHeader
        title="Watchlist"
        description={
          entries.length
            ? `${entries.length} ${entries.length === 1 ? "film" : "films"} saved for later.`
            : undefined
        }
      />

      {watchlist.isLoading ? (
        <div className={GRID}>
          {Array.from({ length: 10 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : watchlist.error ? (
        <EmptyState
          title="Couldn't load your watchlist"
          description="The API didn't respond. Check that the backend is running."
        />
      ) : entries.length === 0 ? (
        <EmptyState
          title="Your watchlist is empty"
          description="Use the bookmark on any poster to save a film here."
          actionText="Browse films"
          actionHref="/"
        />
      ) : (
        <div className={GRID}>
          {entries.map(({ id, movie }) =>
            movie ? (
              <MovieCard
                key={id}
                id={movie.id}
                title={movie.title}
                year={movie.year}
                genres={movie.genres}
                poster_path={movie.poster_path}
                userRating={ratingsById[movie.id] || null}
                isInWatchlist
              />
            ) : null
          )}
        </div>
      )}
    </div>
  );
};
