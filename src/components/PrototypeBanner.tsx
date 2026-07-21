export function PrototypeBanner() {
  return (
    <aside className="prototype-banner" role="status" aria-live="polite">
      <p>
        <strong>Hybrid prototype:</strong> Base Model, RAG, and Fine-Tuned are live (Fine-Tuned
        needs <code>FINETUNED_SERVICE_URL</code>). Fine-Tuned + RAG remain simulated.
      </p>
    </aside>
  );
}
