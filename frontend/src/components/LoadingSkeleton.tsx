import React from "react";
import { Link } from "react-router-dom";

export const CardSkeleton: React.FC = () => (
  <div className="w-full animate-pulse">
    <div className="aspect-[2/3] w-full rounded-lg bg-cinema-850" />
    <div className="mt-2.5 h-3.5 w-3/4 rounded bg-cinema-850" />
    <div className="mt-1.5 h-3 w-1/3 rounded bg-cinema-850" />
  </div>
);

export const RowSkeleton: React.FC<{ count?: number }> = ({ count = 5 }) => (
  <div className="space-y-3">
    <div className="h-5 w-40 animate-pulse rounded bg-cinema-850" />
    <div className="flex gap-4 overflow-hidden">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="w-36 flex-shrink-0 sm:w-44 md:w-48">
          <CardSkeleton />
        </div>
      ))}
    </div>
  </div>
);

export const EmptyState: React.FC<{
  title: string;
  description: string;
  actionText?: string;
  actionHref?: string;
  onAction?: () => void;
}> = ({ title, description, actionText, actionHref, onAction }) => (
  <div className="my-8 rounded-xl border border-dashed border-cinema-750 px-6 py-14 text-center">
    <h3 className="font-display text-base font-semibold text-white">{title}</h3>
    <p className="mx-auto mt-1 max-w-sm text-sm text-cinema-400">{description}</p>
    {actionText && actionHref && (
      <Link to={actionHref} className="btn-primary mt-5">
        {actionText}
      </Link>
    )}
    {actionText && !actionHref && onAction && (
      <button onClick={onAction} className="btn-primary mt-5">
        {actionText}
      </button>
    )}
  </div>
);
