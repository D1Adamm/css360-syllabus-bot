import { PageHeader } from '../components/PageHeader';
import { PlaceholderPanel } from '../components/PlaceholderPanel';

export function SeedDatasetPage() {
  return (
    <>
      <PageHeader
        title="Seed Dataset"
        description="Review and manage the collection of seed examples created for fine-tuning."
      />
      <PlaceholderPanel title="Coming in a future phase">
        The seed dataset view will display all created examples in a table or
        list, with options to filter, edit, and export the dataset for training.
      </PlaceholderPanel>
    </>
  );
}
