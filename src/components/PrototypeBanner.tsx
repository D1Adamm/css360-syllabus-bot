export function PrototypeBanner() {
  return (
    <aside className="prototype-banner" role="status" aria-live="polite">
      <p>
        <strong>Live prototype:</strong> Base Model, RAG, Fine-Tuned, and Fine-Tuned + RAG are
        live (Fine-Tuned paths need <code>FINETUNED_SERVICE_URL</code>).
      </p>
    </aside>
  );
}
