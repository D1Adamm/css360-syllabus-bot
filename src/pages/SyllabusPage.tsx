import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function SyllabusPage() {
  return (
    <>
      <PageHeader
        title="Syllabus Explorer"
        description="Browse and search course syllabus content to identify topics for seed question creation."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The syllabus explorer will load content from the course syllabus and let
        students navigate sections, view topic summaries, and select passages as
        context for building seed examples.
      </PlaceholderPanel>
    </>
  );
}
