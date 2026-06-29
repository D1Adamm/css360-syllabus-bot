import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function ArchitecturePage() {
  return (
    <>
      <PageHeader
        title="Architecture"
        description="Understand how the syllabus fine-tuning classroom prototype is structured technically."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The architecture page will document the system design, data flow between
        syllabus content, seed examples, model inference, and evaluation
        components for student researchers.
      </PlaceholderPanel>
    </>
  );
}
