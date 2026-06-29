import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function EvaluationPage() {
  return (
    <>
      <PageHeader
        title="Evaluation"
        description="Rate and compare model responses on accuracy, helpfulness, conciseness, and grounding."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The evaluation interface will let students score each model response,
        flag potential hallucinations, and record qualitative feedback for
        research analysis.
      </PlaceholderPanel>
    </>
  );
}
