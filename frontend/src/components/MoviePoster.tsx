import React, { useState } from "react";

interface MoviePosterProps {
  title: string;
  year?: number | null;
  genres?: string[];
  poster_path?: string | null;
  className?: string;
  size?: "sm" | "md" | "lg";
}

const genreGradients: [RegExp, string][] = [
  [/sci-fi|fantasy/, "from-slate-950 via-indigo-950 to-cyan-950"],
  [/action|adventure/, "from-stone-950 via-red-950 to-amber-950"],
  [/horror|thriller|mystery/, "from-black via-zinc-950 to-crimson-950"],
  [/animation|children/, "from-violet-950 via-purple-950 to-teal-950"],
  [/comedy|romance/, "from-neutral-950 via-pink-950 to-rose-950"],
  [/drama|crime/, "from-neutral-950 via-emerald-950 to-stone-950"],
];

const gradientFor = (genres: string[]) => {
  const joined = genres.join(" ").toLowerCase();
  return (
    genreGradients.find(([pattern]) => pattern.test(joined))?.[1] ??
    "from-cinema-950 via-cinema-900 to-cinema-850"
  );
};

const titleSizes = { sm: "text-xs", md: "text-sm", lg: "text-base" };

export const MoviePoster: React.FC<MoviePosterProps> = ({
  title,
  year,
  genres = [],
  poster_path,
  className = "",
  size = "md",
}) => {
  // 0 = TMDB CDN, 1 = backend poster proxy (for networks that block TMDB), 2 = text fallback
  const [loadStage, setLoadStage] = useState<0 | 1 | 2>(0);

  const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const isAbsolute = poster_path?.startsWith("http");
  const directUrl = poster_path
    ? isAbsolute
      ? poster_path
      : `https://image.tmdb.org/t/p/w500${poster_path}`
    : null;
  const proxyUrl =
    poster_path && !isAbsolute
      ? `${apiBase}/api/v1/movies/poster-proxy?path=${encodeURIComponent(poster_path)}`
      : null;

  const src = loadStage === 0 ? directUrl : loadStage === 1 ? proxyUrl : null;

  if (src) {
    return (
      <img
        src={src}
        alt={title}
        referrerPolicy="no-referrer"
        crossOrigin="anonymous"
        onError={() => setLoadStage(loadStage === 0 && proxyUrl ? 1 : 2)}
        className={`h-full w-full object-cover ${className}`}
        loading="lazy"
      />
    );
  }

  return (
    <div
      className={`flex h-full w-full select-none flex-col justify-end bg-gradient-to-br p-3 ${gradientFor(genres)} ${className}`}
    >
      <p
        className={`line-clamp-3 font-display font-semibold leading-tight text-white ${titleSizes[size]}`}
      >
        {title}
      </p>
      {year && <p className="mt-0.5 text-[11px] text-cinema-400">{year}</p>}
    </div>
  );
};
