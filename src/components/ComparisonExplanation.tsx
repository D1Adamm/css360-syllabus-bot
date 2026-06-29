interface ComparisonExplanationProps {
  notes: string;
}

export function ComparisonExplanation({ notes }: ComparisonExplanationProps) {
  return (
    <section className="comparison-explanation" aria-labelledby="comparison-explanation-title">
      <h2 id="comparison-explanation-title" className="comparison-explanation__title">
        Why these responses differ
      </h2>
      <p className="comparison-explanation__text">{notes}</p>
      <p className="comparison-explanation__annotation-note">
        Grounding labels (Low, Medium, High) are prototype annotations that describe how
        closely each response appears tied to syllabus facts. They are not objective
        measured scores.
      </p>
    </section>
  );
}
