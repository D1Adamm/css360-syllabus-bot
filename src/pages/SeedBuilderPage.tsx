import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function SeedBuilderPage() {
  return (
    <>
      <PageHeader
        title="Seed Data Builder"
        description="Create question-and-answer examples from syllabus content to use in fine-tuning experiments."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The seed data builder will provide a form for writing instruction-response
        pairs, tagging categories and difficulty levels, and linking each example
        to a specific syllabus section.
      </PlaceholderPanel>
    </>
  );
}
