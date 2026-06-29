import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function ComparisonPage() {
  return (
    <>
      <PageHeader
        title="Model Comparison"
        description="Compare responses from base, RAG, fine-tuned, and fine-tuned plus RAG models side by side."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The model comparison view will present the same question to four model
        configurations and display their responses with grounding indicators for
        classroom discussion.
      </PlaceholderPanel>
    </>
  );
}
