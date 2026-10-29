import React from "react";
import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { deleteRating, getMyRatings, updateRating } from "../api/ratings";
import { StarRating } from "../components/StarRating";
import { MoviePoster } from "../components/MoviePoster";
import { PageHeader } from "../components/PageHeader";
import { EmptyState } from "../components/LoadingSkeleton";

export const RatingsPage: React.FC = () => {
  const queryClient = useQueryClient();

  const {
    data: ratings,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["ratings"],
    queryFn: getMyRatings,
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["ratings"] });
    queryClient.invalidateQueries({ queryKey: ["recommendations"] });
  };

  const deleteMutation = useMutation({
    mutationFn: (movieId: number) => deleteRating(movieId),
    onSuccess: invalidate,
  });

  const updateMutation = useMutation({
    mutationFn: ({ movieId, rating }: { movieId: number; rating: number }) =>
      updateRating(movieId, rating),
    onSuccess: invalidate,
  });

  return (
    <div className="mx-auto max-w-5xl space-y-8 px-4 py-10 sm:px-6 lg:px-8">
      <PageHeader
        title="Your ratings"
        description={
          ratings?.length
            ? `${ratings.length} ${ratings.length === 1 ? "film" : "films"} rated. Changing a rating updates your recommendations.`
            : undefined
        }
      />

      {isLoading ? (
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-20 animate-pulse rounded-lg bg-cinema-850" />
          ))}
        </div>
      ) : error ? (
        <EmptyState
          title="Couldn't load ratings"
          description="The API didn't respond. Check that the backend is running."
        />
      ) : !ratings || ratings.length === 0 ? (
        <EmptyState
          title="No ratings yet"
          description="Rate movies as you browse, or set up your taste profile to get started."
          actionText="Set up taste profile"
          actionHref="/onboarding"
        />
      ) : (
        <ul className="divide-y divide-cinema-800 border-y border-cinema-800">
          {ratings.map((entry) => {
            const movie = entry.movie;
            return (
              <li key={entry.id} className="flex items-center gap-4 py-3">
                <Link
                  to={`/movies/${entry.movie_id}`}
                  className="h-16 w-11 shrink-0 overflow-hidden rounded"
                >
                  <MoviePoster
                    title={movie?.title || "Movie"}
                    year={movie?.year}
                    genres={movie?.genres}
                    poster_path={movie?.poster_path}
                    size="sm"
                  />
                </Link>

                <div className="min-w-0 flex-1">
                  <Link
                    to={`/movies/${entry.movie_id}`}
                    className="line-clamp-1 text-sm font-medium text-white hover:text-crimson-400"
                  >
                    {movie?.title || `Movie #${entry.movie_id}`}
                  </Link>
                  <p className="text-xs text-cinema-500">
                    {[movie?.year, `rated ${new Date(entry.created_at).toLocaleDateString()}`]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                </div>

                <StarRating
                  rating={entry.rating}
                  size="sm"
                  readOnly={false}
                  onChange={(rating) => updateMutation.mutate({ movieId: entry.movie_id, rating })}
                  showNumeric
                />

                <button
                  onClick={() => deleteMutation.mutate(entry.movie_id)}
                  disabled={deleteMutation.isPending}
                  className="rounded-md p-1.5 text-cinema-500 hover:bg-cinema-850 hover:text-crimson-400"
                  aria-label="Remove rating"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};
