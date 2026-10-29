import React from "react";

interface PageHeaderProps {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({ title, description, actions }) => (
  <header className="flex flex-col gap-4 border-b border-cinema-800 pb-6 sm:flex-row sm:items-end sm:justify-between">
    <div>
      <h1 className="font-display text-2xl font-bold tracking-tight text-white sm:text-3xl">
        {title}
      </h1>
      {description && <p className="mt-1.5 max-w-2xl text-sm text-cinema-400">{description}</p>}
    </div>
    {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
  </header>
);
