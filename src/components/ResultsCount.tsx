interface ResultsCountProps {
  resultCount: number;
  totalCount: number;
}

export function ResultsCount({ resultCount, totalCount }: ResultsCountProps) {
  return (
    <p className="results-count" aria-live="polite">
      Showing {resultCount} of {totalCount} seed examples
    </p>
  );
}
