interface PlaceholderPanelProps {
  title: string;
  children: React.ReactNode;
}

export function PlaceholderPanel({ title, children }: PlaceholderPanelProps) {
  return (
    <section className="placeholder-panel" aria-label={title}>
      <h2 className="placeholder-panel__title">{title}</h2>
      <p className="placeholder-panel__text">{children}</p>
    </section>
  );
}
