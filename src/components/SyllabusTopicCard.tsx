import { useId, useState } from 'react';
import type { SyllabusTopic } from '../types';

interface SyllabusTopicCardProps {
  topic: SyllabusTopic;
}

export function SyllabusTopicCard({ topic }: SyllabusTopicCardProps) {
  const [expanded, setExpanded] = useState(false);
  const detailsId = useId();

  return (
    <article className="syllabus-topic-card">
      <div className="syllabus-topic-card__header">
        <span className="syllabus-topic-card__category">{topic.category}</span>
        <h2 className="syllabus-topic-card__title">{topic.title}</h2>
      </div>
      <p className="syllabus-topic-card__summary">{topic.summary}</p>
      <p className="syllabus-topic-card__source">
        <span className="syllabus-topic-card__source-label">Source section:</span>{' '}
        {topic.sourceSection}
      </p>
      <button
        type="button"
        className="syllabus-topic-card__toggle"
        aria-expanded={expanded}
        aria-controls={detailsId}
        onClick={() => setExpanded((current) => !current)}
      >
        {expanded ? 'Hide details' : 'View details'}
      </button>
      {expanded && (
        <div id={detailsId} className="syllabus-topic-card__details">
          <h3 className="syllabus-topic-card__details-heading">Key details</h3>
          <ul className="syllabus-topic-card__details-list">
            {topic.details.map(
              (detail) => (
                <li key={detail}>{detail}</li>
              ),
            )}
          </ul>
        </div>
      )}
    </article>
  );
}
