import { useQuery } from "@tanstack/react-query";
import { getMyRatings } from "../api/ratings";
import { getWatchlist } from "../api/watchlist";
import { useAuth } from "../context/AuthContext";

/** The current user's ratings and watchlist, indexed by movie id. */
export function useLibrary() {
  const { isAuthenticated } = useAuth();

  const ratings = useQuery({
    queryKey: ["ratings"],
    queryFn: getMyRatings,
    enabled: isAuthenticated,
  });

  const watchlist = useQuery({
    queryKey: ["watchlist"],
    queryFn: getWatchlist,
    enabled: isAuthenticated,
  });

  const ratingsById: Record<number, number> = {};
  for (const r of ratings.data ?? []) ratingsById[r.movie_id] = r.rating;

  const watchlistIds: Record<number, boolean> = {};
  for (const w of watchlist.data ?? []) watchlistIds[w.movie_id] = true;

  return { ratings, watchlist, ratingsById, watchlistIds };
}
