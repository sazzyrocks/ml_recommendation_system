import React, { useState } from "react";
import { Star } from "lucide-react";

interface StarRatingProps {
  rating: number; // 0 to 5
  size?: "sm" | "md" | "lg";
  readOnly?: boolean;
  onChange?: (newRating: number) => void;
  showNumeric?: boolean;
}

const starSizes = {
  sm: "w-3.5 h-3.5",
  md: "w-4 h-4",
  lg: "w-6 h-6",
};

export const StarRating: React.FC<StarRatingProps> = ({
  rating,
  size = "md",
  readOnly = true,
  onChange,
  showNumeric = false,
}) => {
  const [hoverRating, setHoverRating] = useState<number | null>(null);
  const activeRating = hoverRating ?? rating;

  return (
    <div className="flex items-center gap-1.5">
      <div className="flex items-center" onMouseLeave={() => setHoverRating(null)}>
        {[1, 2, 3, 4, 5].map((star) => {
          const isFilled = activeRating >= star;
          const isHalf = !isFilled && activeRating >= star - 0.5;

          return (
            <button
              key={star}
              type="button"
              disabled={readOnly}
              onClick={() => onChange?.(star)}
              onMouseEnter={() => !readOnly && setHoverRating(star)}
              aria-label={`${star} of 5`}
              className={`p-0.5 focus:outline-none ${readOnly ? "cursor-default" : "cursor-pointer"}`}
            >
              <Star
                className={`${starSizes[size]} ${
                  isFilled
                    ? "fill-gold-400 text-gold-400"
                    : isHalf
                      ? "fill-gold-400/50 text-gold-400"
                      : "fill-transparent text-cinema-600"
                }`}
              />
            </button>
          );
        })}
      </div>
      {showNumeric && (
        <span className="text-xs font-medium tabular-nums text-gold-400">
          {rating > 0 ? rating.toFixed(1) : "—"}
        </span>
      )}
    </div>
  );
};
