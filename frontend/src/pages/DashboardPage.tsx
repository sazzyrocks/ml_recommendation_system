import React from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { getMyRecommendations } from "../api/recommendations";
import { MovieCard } from "../components/MovieCard";
import { PageHeader } from "../components/PageHeader";
import { CardSkeleton, EmptyState } from "../components/LoadingSkeleton";
import { useLibrary } from "../hooks/useLibrary";

const GRID = "grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5";

export const DashboardPage: React.FC = () => {
  const { ratings, ratingsById, watchlistIds } = useLibrary();

  const { data, isLoading, isRefetching, refetch, error } = useQuery({
    queryKey: ["recommendations"],
    queryFn: () => getMyRecommendations(20),
    staleTime: 1000 * 60 * 5,
  });

  const items = data?.items || [];
  const ratedCount = ratings.data?.length ?? 0;

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6 lg:px-8">
      <PageHeader
        title="For you"
        description={
          ratedCount > 0
            ? `Based on the ${ratedCount} ${ratedCount === 1 ? "film" : "films"} you've rated. Rate more and this list updates straight away.`
            : "Rate a few films and this list will start to reflect your taste."
        }
        actions={
          <>
            <button onClick={() => refetch()} disabled={isRefetching} className="btn-secondary">
              <RefreshCw className={`h-3.5 w-3.5 ${isRefetching ? "animate-spin" : ""}`} />
              {isRefetching ? "Refreshing" : "Refresh"}
            </button>
            <Link to="/onboarding" className="btn-primary">
              Edit taste profile
            </Link>
          </>
        }
      />

      {isLoading ? (
        <div className={GRID}>
          {Array.from({ length: 15 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : error ? (
        <EmptyState
          title="Couldn't load recommendations"
          description="The API didn't respond. Check that the backend is running."
          actionText="Try again"
          onAction={() => refetch()}
        />
      ) : items.length === 0 ? (
        <EmptyState
          title="Nothing here yet"
          description="Rate a few movies or set up your taste profile to get recommendations."
          actionText="Set up taste profile"
          actionHref="/onboarding"
        />
      ) : (
        <div className={GRID}>
          {items.map((movie) => (
            <MovieCard
              key={movie.movie_id}
              id={movie.movie_id}
              title={movie.title}
              year={movie.year}
              genres={movie.genres}
              poster_path={movie.poster_path}
              userRating={ratingsById[movie.movie_id] || null}
              isInWatchlist={!!watchlistIds[movie.movie_id]}
              reason={movie.reason}
              topSource={movie.top_source}
            />
          ))}
        </div>
      )}
    </div>
  );
};
