import type { DatasetStatistics } from '../utils/seedDataUtils';

interface DatasetStatsProps {
  stats: DatasetStatistics;
}

export function DatasetStats({ stats }: DatasetStatsProps) {
  const items = [
    { label: 'Total examples', value: stats.totalExamples },
    { label: 'Categories', value: stats.totalCategories },
    { label: 'Easy', value: stats.easyCount },
    { label: 'Medium', value: stats.mediumCount },
    { label: 'Hard', value: stats.hardCount },
    { label: 'Directly answered', value: stats.directlyAnsweredCount },
    { label: 'Requires clarification', value: stats.notDirectlyAnsweredCount },
  ];

  return (
    <section className="dataset-stats" aria-labelledby="dataset-stats-title">
      <h2 id="dataset-stats-title" className="dataset-stats__title">
        Dataset statistics
      </h2>
      <ul className="dataset-stats__grid">
        {items.map((item) => (
          <li key={item.label} className="dataset-stats__card">
            <span className="dataset-stats__value">{item.value}</span>
            <span className="dataset-stats__label">{item.label}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
