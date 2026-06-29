interface SectionCardProps {
  title: string;
  children: React.ReactNode;
}

export function SectionCard({ title, children }: SectionCardProps) {
  return (
    <article className="section-card">
      <h3 className="section-card__title">{title}</h3>
      <p className="section-card__text">{children}</p>
    </article>
  );
}
