import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Search, X, Sparkles, Film, ArrowRight, Loader2 } from "lucide-react";
import { getMovies, searchMovies } from "../api/movies";
import { createRating } from "../api/ratings";
import { StarRating } from "../components/StarRating";
import { MoviePoster } from "../components/MoviePoster";
import { PageHeader } from "../components/PageHeader";
import { useAuth } from "../context/AuthContext";

const GENRES = [
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Drama",
  "Fantasy",
  "Horror",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "Western",
];

const MIN_GENRES = 3;
const MIN_RATINGS = 5;

type DisplayMovie = {
  id: number;
  title: string;
  year?: number | null;
  genres: string[];
  poster_path?: string | null;
};

export const OnboardingPage: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [ratedMovies, setRatedMovies] = useState<Record<number, DisplayMovie>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Search & browse controls
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [activeGenreFilter, setActiveGenreFilter] = useState<string>("All");
  const [page, setPage] = useState(1);
  const [accumulatedMovies, setAccumulatedMovies] = useState<DisplayMovie[]>([]);

  // Debounce search query
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedQuery(searchQuery.trim());
    }, 280);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  // Browse movies query (when not searching)
  const { data: browseData, isLoading: isBrowseLoading } = useQuery({
    queryKey: ["onboarding", "browse", activeGenreFilter, page],
    queryFn: () =>
      getMovies({
        page,
        limit: 18,
        sort: "popular",
        genre: activeGenreFilter !== "All" ? activeGenreFilter : undefined,
      }),
    enabled: debouncedQuery.length === 0,
  });

  // Search movies query (when typing in search box)
  const { data: searchResults, isLoading: isSearchLoading } = useQuery({
    queryKey: ["onboarding", "search", debouncedQuery],
    queryFn: () => searchMovies({ q: debouncedQuery, k: 18 }),
    enabled: debouncedQuery.length > 0,
  });

  // Accumulate browsed movies so "Load more" appends seamlessly
  useEffect(() => {
    if (browseData?.items) {
      if (page === 1) {
        setAccumulatedMovies(browseData.items);
      } else {
        setAccumulatedMovies((prev) => {
          const existingIds = new Set(prev.map((m) => m.id));
          const newItems = browseData.items.filter((m) => !existingIds.has(m.id));
          return [...prev, ...newItems];
        });
      }
    }
  }, [browseData, page]);

  // Reset page when genre filter changes
  const handleGenreFilterChange = (genre: string) => {
    setActiveGenreFilter(genre);
    setPage(1);
    setSearchQuery("");
  };

  const toggleGenre = (genre: string) =>
    setSelectedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]
    );

  const handleRateMovie = (movie: DisplayMovie, stars: number) => {
    setRatings((prev) => ({ ...prev, [movie.id]: stars }));
    setRatedMovies((prev) => ({ ...prev, [movie.id]: movie }));
  };

  const handleRemoveRating = (movieId: number) => {
    setRatings((prev) => {
      const copy = { ...prev };
      delete copy[movieId];
      return copy;
    });
    setRatedMovies((prev) => {
      const copy = { ...prev };
      delete copy[movieId];
      return copy;
    });
  };

  const ratedCount = Object.keys(ratings).length;
  const canSubmit = selectedGenres.length >= MIN_GENRES && ratedCount >= MIN_RATINGS;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setIsSubmitting(true);
    try {
      if (isAuthenticated) {
        await Promise.all(
          Object.entries(ratings).map(([movieId, stars]) =>
            createRating(Number(movieId), stars).catch(() => null)
          )
        );
        queryClient.invalidateQueries({ queryKey: ["ratings"] });
        queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      }
    } finally {
      setIsSubmitting(false);
      navigate("/dashboard");
    }
  };

  // Determine which movies to display
  const isSearching = debouncedQuery.length > 0;
  const displayMovies: DisplayMovie[] = isSearching ? searchResults || [] : accumulatedMovies;
  const isDisplayLoading = isSearching ? isSearchLoading : isBrowseLoading && page === 1;

  return (
    <div className="mx-auto max-w-5xl space-y-10 px-4 py-10 sm:px-6">
      <PageHeader
        title="Your taste profile"
        description={`Pick at least ${MIN_GENRES} genres you love and rate ${MIN_RATINGS} movies you've seen. You can choose from popular titles, filter by genre, or search any movie in our entire catalog!`}
      />

      {/* Step 1: Genres */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <span>Step 1: Genres you enjoy</span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                selectedGenres.length >= MIN_GENRES
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-cinema-800 text-cinema-400"
              }`}
            >
              {selectedGenres.length}/{MIN_GENRES} selected
            </span>
          </h2>
        </div>
        <div className="flex flex-wrap gap-2">
          {GENRES.map((genre) => {
            const selected = selectedGenres.includes(genre);
            return (
              <button
                key={genre}
                type="button"
                onClick={() => toggleGenre(genre)}
                aria-pressed={selected}
                className={`inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-sm font-medium transition-all ${
                  selected
                    ? "border-crimson-500 bg-crimson-600 text-white shadow-lg shadow-crimson-600/30"
                    : "border-cinema-700 bg-cinema-850/60 text-cinema-300 hover:border-cinema-500 hover:text-white"
                }`}
              >
                {selected && <Check className="h-3.5 w-3.5" />}
                {genre}
              </button>
            );
          })}
        </div>
      </section>

      {/* Step 2: Rate Films (Custom Choice & Search) */}
      <section className="space-y-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <span>Step 2: Rate movies of your choice</span>
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${
                  ratedCount >= MIN_RATINGS
                    ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    : "bg-cinema-800 text-cinema-400"
                }`}
              >
                {ratedCount}/{MIN_RATINGS} rated
              </span>
            </h2>
            <p className="text-xs text-cinema-400 mt-0.5">
              Pick popular hits below or type any movie title in the search box.
            </p>
          </div>
        </div>

        {/* Live Search Bar */}
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-cinema-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search any movie you've seen (e.g. Inception, Avatar, Titanic, Spider-Man)..."
            className="w-full rounded-xl border border-cinema-700 bg-cinema-900/90 pl-10 pr-10 py-3 text-sm text-white placeholder-cinema-500 focus:border-crimson-500 focus:outline-none focus:ring-1 focus:ring-crimson-500 transition-all shadow-inner"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 text-cinema-400 hover:text-white p-0.5 rounded-full hover:bg-cinema-800 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Genre Filter Pills (Only shown when not actively searching) */}
        {!isSearching && (
          <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin">
            <span className="text-xs text-cinema-400 font-medium shrink-0">Filter by:</span>
            {[
              "All",
              ...(selectedGenres.length > 0
                ? selectedGenres
                : ["Action", "Sci-Fi", "Drama", "Comedy", "Thriller"]),
            ].map((g) => {
              const isActive = activeGenreFilter === g;
              return (
                <button
                  key={g}
                  onClick={() => handleGenreFilterChange(g)}
                  className={`shrink-0 rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-crimson-500/20 text-crimson-300 border border-crimson-500/40"
                      : "bg-cinema-850 text-cinema-400 hover:text-white hover:bg-cinema-800 border border-cinema-800"
                  }`}
                >
                  {g === "All" ? "🔥 Most Popular" : g}
                </button>
              );
            })}
          </div>
        )}

        {/* Your Rated Films Tray (if user has rated any) */}
        {ratedCount > 0 && (
          <div className="rounded-xl border border-gold-500/30 bg-cinema-900/80 p-4 shadow-lg backdrop-blur">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-gold-400 flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5" />
                Your Rated Movies ({ratedCount})
              </span>
              <span className="text-xs text-cinema-400">Click stars to update rating</span>
            </div>
            <div className="grid gap-2.5 sm:grid-cols-2 md:grid-cols-3">
              {Object.entries(ratings).map(([idStr, stars]) => {
                const movieId = Number(idStr);
                const movie = ratedMovies[movieId];
                if (!movie) return null;
                return (
                  <div
                    key={movieId}
                    className="flex items-center justify-between gap-2.5 rounded-lg border border-gold-500/20 bg-cinema-850/80 p-2 text-xs"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium text-white">{movie.title}</p>
                      <p className="text-[10px] text-cinema-400">
                        {movie.year ? `${movie.year} · ` : ""}
                        {stars} ★
                      </p>
                    </div>
                    <StarRating
                      rating={stars}
                      size="sm"
                      readOnly={false}
                      onChange={(newStars) => handleRateMovie(movie, newStars)}
                    />
                    <button
                      onClick={() => handleRemoveRating(movieId)}
                      title="Remove rating"
                      className="text-cinema-500 hover:text-red-400 p-1"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Movies Grid */}
        {isDisplayLoading ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 9 }).map((_, i) => (
              <div key={i} className="h-28 animate-pulse rounded-lg bg-cinema-850" />
            ))}
          </div>
        ) : displayMovies.length === 0 ? (
          <div className="rounded-xl border border-dashed border-cinema-700 py-12 text-center">
            <Film className="mx-auto h-8 w-8 text-cinema-500 mb-2" />
            <p className="text-sm font-medium text-white">No movies found</p>
            <p className="text-xs text-cinema-400 mt-1">
              {isSearching
                ? `No title matched "${debouncedQuery}". Try another spelling or title.`
                : "Try selecting a different genre filter above."}
            </p>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {displayMovies.map((movie) => {
              const rating = ratings[movie.id] || 0;
              return (
                <div
                  key={movie.id}
                  className={`flex gap-3 rounded-lg border p-2.5 transition-all ${
                    rating > 0
                      ? "border-gold-500/60 bg-cinema-900/90 shadow-md shadow-gold-500/5 ring-1 ring-gold-500/30"
                      : "border-cinema-800 bg-cinema-850/40 hover:border-cinema-700 hover:bg-cinema-850/80"
                  }`}
                >
                  <div className="h-24 w-16 shrink-0 overflow-hidden rounded bg-cinema-950">
                    <MoviePoster
                      title={movie.title}
                      year={movie.year}
                      genres={movie.genres}
                      poster_path={movie.poster_path}
                      size="sm"
                    />
                  </div>
                  <div className="flex min-w-0 flex-1 flex-col justify-between py-0.5">
                    <div>
                      <p className="line-clamp-2 text-sm font-medium leading-tight text-white">
                        {movie.title}
                      </p>
                      <p className="mt-0.5 text-xs text-cinema-500">
                        {[movie.year, movie.genres?.[0]].filter(Boolean).join(" · ")}
                      </p>
                    </div>
                    <div className="pt-2">
                      <StarRating
                        rating={rating}
                        size="sm"
                        readOnly={false}
                        onChange={(stars) =>
                          handleRateMovie(
                            {
                              id: movie.id,
                              title: movie.title,
                              year: movie.year,
                              genres: movie.genres || [],
                              poster_path: movie.poster_path,
                            },
                            stars
                          )
                        }
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Load More Button (When browsing) */}
        {!isSearching && (
          <div className="flex justify-center pt-2">
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={isBrowseLoading}
              className="inline-flex items-center gap-2 rounded-xl border border-cinema-700 bg-cinema-850 px-5 py-2.5 text-xs font-medium text-cinema-200 hover:border-cinema-500 hover:text-white transition-colors"
            >
              {isBrowseLoading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Loading more movies...
                </>
              ) : (
                <>
                  <Film className="h-3.5 w-3.5 text-cinema-400" />
                  Load More Movies to Choose From
                </>
              )}
            </button>
          </div>
        )}
      </section>

      {/* Sticky Bottom Actions */}
      <div className="sticky bottom-4 z-40 flex items-center justify-between gap-4 rounded-xl border border-cinema-700 bg-cinema-900/95 px-5 py-3.5 backdrop-blur-md shadow-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center sm:gap-3">
          <span className="text-sm font-medium text-white">
            {canSubmit ? "Ready to generate your recommendations!" : "Profile progress:"}
          </span>
          <span className="text-xs text-cinema-400">
            {canSubmit
              ? `${selectedGenres.length} genres · ${ratedCount} movies rated`
              : `${Math.max(0, MIN_GENRES - selectedGenres.length)} more genres, ${Math.max(0, MIN_RATINGS - ratedCount)} more ratings needed`}
          </span>
        </div>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit || isSubmitting}
          className="btn-primary inline-flex items-center gap-2 px-5 py-2.5 text-sm"
        >
          <span>{isSubmitting ? "Personalizing…" : "Show recommendations"}</span>
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
};
