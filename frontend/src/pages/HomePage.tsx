import React from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getMovies } from "../api/movies";
import { MovieRow } from "../components/MovieRow";
import { RowSkeleton } from "../components/LoadingSkeleton";
import { useLibrary } from "../hooks/useLibrary";

interface CatalogRowConfig {
  key: string;
  title: string;
  subtitle?: string;
  genre?: string;
}

const ROWS: CatalogRowConfig[] = [
  { key: "trending", title: "Popular right now", subtitle: "Most-rated titles in the catalog" },
  { key: "scifi", genre: "Sci-Fi", title: "Science fiction" },
  { key: "action", genre: "Action", title: "Action" },
  { key: "animation", genre: "Animation", title: "Animation" },
  { key: "drama", genre: "Drama", title: "Drama" },
];

const CatalogRow: React.FC<{
  row: CatalogRowConfig;
  ratingsById: Record<number, number>;
  watchlistIds: Record<number, boolean>;
}> = ({ row, ratingsById, watchlistIds }) => {
  const { data, isLoading } = useQuery({
    queryKey: ["movies", row.key],
    queryFn: () =>
      getMovies({ genre: row.genre, limit: row.genre ? 10 : 12, sort: "popular", page: 1 }),
  });

  if (isLoading) return <RowSkeleton count={5} />;

  return (
    <MovieRow
      title={row.title}
      subtitle={row.subtitle}
      movies={data?.items || []}
      userRatingsMap={ratingsById}
      watchlistMap={watchlistIds}
    />
  );
};

export const HomePage: React.FC = () => {
  const { ratingsById, watchlistIds } = useLibrary();

  return (
    <div className="mx-auto max-w-7xl space-y-12 px-4 py-10 sm:px-6 lg:px-8">
      <section className="max-w-2xl">
        <h1 className="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Find something worth watching tonight.
        </h1>
        <p className="mt-3 text-cinema-300">
          Rate a handful of films you know and CineMatch suggests what to watch next, with a short
          note on why each pick showed up.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/dashboard" className="btn-primary">
            See my recommendations
          </Link>
          <Link to="/search" className="btn-secondary">
            Search by plot or mood
          </Link>
        </div>
      </section>

      {ROWS.map((row) => (
        <CatalogRow key={row.key} row={row} ratingsById={ratingsById} watchlistIds={watchlistIds} />
      ))}
    </div>
  );
};
