import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function ResultsPage() {
  return (
    <>
      <PageHeader
        title="Results"
        description="View aggregated evaluation results and classroom findings across model approaches."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The results dashboard will summarize evaluation scores, highlight
        preferred models per question category, and present trends from
        classroom evaluation sessions.
      </PlaceholderPanel>
    </>
  );
}
