export function PrototypeBanner() {
  return (
    <aside className="prototype-banner" role="status" aria-live="polite">
      <p>
        <strong>Hybrid prototype:</strong> Base Model and RAG are live locally. Fine-Tuned and
        Fine-Tuned + RAG remain simulated.
      </p>
    </aside>
  );
}
