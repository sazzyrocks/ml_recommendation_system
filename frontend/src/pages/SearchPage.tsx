import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search as SearchIcon, X } from "lucide-react";
import { searchMovies } from "../api/movies";
import { MovieCard } from "../components/MovieCard";
import { PageHeader } from "../components/PageHeader";
import { CardSkeleton, EmptyState } from "../components/LoadingSkeleton";
import { useLibrary } from "../hooks/useLibrary";

const EXAMPLE_QUERIES = [
  "space exploration with a father-daughter story",
  "lighthearted heist comedy",
  "dark psychological thriller in a rainy city",
  "cyberpunk detective mystery",
  "coming-of-age road trip",
];

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
];

const GRID = "grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6";

export const SearchPage: React.FC = () => {
  const { ratingsById, watchlistIds } = useLibrary();
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [yearMin, setYearMin] = useState("");
  const [yearMax, setYearMax] = useState("");

  const { data: results, isFetching } = useQuery({
    queryKey: ["search", activeQuery, genre, yearMin, yearMax],
    queryFn: () =>
      searchMovies({
        q: activeQuery,
        k: 24,
        genre: genre || undefined,
        year_min: yearMin ? parseInt(yearMin) : undefined,
        year_max: yearMax ? parseInt(yearMax) : undefined,
      }),
    enabled: activeQuery.trim().length > 0,
  });

  const runSearch = (q: string) => {
    setQuery(q);
    setActiveQuery(q.trim());
  };

  const hasFilters = Boolean(genre || yearMin || yearMax);
  const clearFilters = () => {
    setGenre("");
    setYearMin("");
    setYearMax("");
  };

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-10 sm:px-6 lg:px-8">
      <PageHeader
        title="Search"
        description="Describe the kind of film you're after in plain words. Results are matched on plot and theme, not just titles."
      />

      <div className="max-w-3xl space-y-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (query.trim()) runSearch(query);
          }}
          className="relative flex items-center"
        >
          <SearchIcon className="pointer-events-none absolute left-3.5 h-4 w-4 text-cinema-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. slow-burn mystery set in a small town"
            className="field py-3 pl-10 pr-28 text-sm"
          />
          <div className="absolute right-1.5 flex items-center gap-1">
            {query && (
              <button
                type="button"
                onClick={() => setQuery("")}
                className="p-2 text-cinema-500 hover:text-white"
                aria-label="Clear"
              >
                <X className="h-4 w-4" />
              </button>
            )}
            <button type="submit" className="btn-primary py-1.5">
              Search
            </button>
          </div>
        </form>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-cinema-500">
          <span>Try:</span>
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => runSearch(example)}
              className={`underline-offset-2 hover:underline ${
                activeQuery === example ? "text-crimson-400" : "text-cinema-300"
              }`}
            >
              {example}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-end gap-3 text-xs">
          <label className="w-40">
            <span className="mb-1 block text-cinema-500">Genre</span>
            <select value={genre} onChange={(e) => setGenre(e.target.value)} className="field">
              <option value="">Any</option>
              {GENRES.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </label>
          <label className="w-28">
            <span className="mb-1 block text-cinema-500">From year</span>
            <input
              type="number"
              placeholder="1990"
              value={yearMin}
              onChange={(e) => setYearMin(e.target.value)}
              className="field"
            />
          </label>
          <label className="w-28">
            <span className="mb-1 block text-cinema-500">To year</span>
            <input
              type="number"
              placeholder="2024"
              value={yearMax}
              onChange={(e) => setYearMax(e.target.value)}
              className="field"
            />
          </label>
          {hasFilters && (
            <button
              type="button"
              onClick={clearFilters}
              className="pb-2 text-crimson-400 hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {activeQuery && results && !isFetching && (
        <p className="text-sm text-cinema-400">
          {results.length} {results.length === 1 ? "result" : "results"} for{" "}
          <span className="text-white">"{activeQuery}"</span>
        </p>
      )}

      {isFetching ? (
        <div className={GRID}>
          {Array.from({ length: 12 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : activeQuery && results?.length === 0 ? (
        <EmptyState
          title="No matches"
          description="Try a broader description or loosen the genre and year filters."
          actionText={hasFilters ? "Clear filters" : undefined}
          onAction={clearFilters}
        />
      ) : results && results.length > 0 ? (
        <div className={GRID}>
          {results.map((hit) => (
            <MovieCard
              key={hit.id}
              id={hit.id}
              title={hit.title}
              year={hit.year}
              genres={hit.genres}
              poster_path={hit.poster_path}
              similarityScore={hit.similarity}
              userRating={ratingsById[hit.id] || null}
              isInWatchlist={!!watchlistIds[hit.id]}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
};
