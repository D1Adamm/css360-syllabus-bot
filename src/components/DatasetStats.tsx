import type { DatasetStatistics } from '../utils/seedDataUtils';

interface DatasetStatsProps {
  stats: DatasetStatistics;
}

export function DatasetStats({ stats }: DatasetStatsProps) {
  const items = [
    { label: 'Total stored', value: stats.totalExamples },
    { label: 'Approved', value: stats.approvedCount },
    { label: 'Rejected', value: stats.rejectedCount },
    { label: 'Generated', value: stats.generatedCount },
    { label: 'Edited', value: stats.editedCount },
  ];

  return (
    <section className="dataset-stats" aria-labelledby="dataset-stats-title">
      <h2 id="dataset-stats-title" className="dataset-stats__title">
        Review status summary
      </h2>
      <ul className="dataset-stats__grid">
        {items.map((item) => (
          <li key={item.label} className="dataset-stats__card">
            <span className="dataset-stats__value">{item.value}</span>
            <span className="dataset-stats__label">{item.label}</span>
          </li>
        ))}
      </ul>
      <p className="dataset-stats__note">
        Approved examples are fine-tuning-ready. Rejected and generated seeds stay stored
        and are excluded from the default Approved view.
      </p>
    </section>
  );
}
