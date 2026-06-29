import { useMemo, useState } from 'react';
import syllabusTopicsData from '../data/syllabusTopics.json';
import { ALL_CATEGORIES, CategoryFilter } from '../components/CategoryFilter';
import { PageHeader } from '../components/PageHeader';
import { SyllabusSearch } from '../components/SyllabusSearch';
import { SyllabusTopicCard } from '../components/SyllabusTopicCard';
import type { SyllabusTopic } from '../types';

const syllabusTopics = syllabusTopicsData as SyllabusTopic[];

function normalizeSearchText(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

function topicMatchesSearch(topic: SyllabusTopic, query: string): boolean {
  if (!query) {
    return true;
  }

  const searchableText = [
    topic.title,
    topic.summary,
    topic.category,
    topic.sourceSection,
    ...topic.details,
  ]
    .join(' ')
    .toLowerCase();

  return searchableText.includes(query);
}

export function SyllabusPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORIES);

  const categories = useMemo(() => {
    const uniqueCategories = new Set(syllabusTopics.map((topic) => topic.category));
    return Array.from(uniqueCategories).sort((left, right) => left.localeCompare(right));
  }, []);

  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {
      [ALL_CATEGORIES]: syllabusTopics.length,
    };

    for (const category of categories) {
      counts[category] = syllabusTopics.filter((topic) => topic.category === category).length;
    }

    return counts;
  }, [categories]);

  const normalizedQuery = normalizeSearchText(searchQuery);

  const filteredTopics = useMemo(() => {
    return syllabusTopics.filter((topic) => {
      const matchesCategory =
        selectedCategory === ALL_CATEGORIES || topic.category === selectedCategory;
      const matchesSearch = topicMatchesSearch(topic, normalizedQuery);

      return matchesCategory && matchesSearch;
    });
  }, [normalizedQuery, selectedCategory]);

  function clearFilters() {
    setSearchQuery('');
    setSelectedCategory(ALL_CATEGORIES);
  }

  return (
    <>
      <PageHeader
        title="Syllabus Explorer"
        description="Browse searchable course policies and expectations organized by topic to help you find answers about CSS360."
      />

      <aside className="syllabus-notice" aria-label="Syllabus source note">
        <p>
          <strong>Prototype note:</strong> Topic summaries are drawn from{' '}
          <code>docs/syllabus.txt</code>. Always consult the official syllabus on Canvas
          and weekly announcements for the most current details.
        </p>
      </aside>

      <section className="syllabus-overview" aria-labelledby="category-overview-title">
        <h2 id="category-overview-title" className="syllabus-overview__title">
          Topics by category
        </h2>
        <ul className="syllabus-overview__list">
          {categories.map((category) => (
            <li key={category} className="syllabus-overview__item">
              <span className="syllabus-overview__category">{category}</span>
              <span className="syllabus-overview__count">{categoryCounts[category]}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="syllabus-controls" aria-label="Search and filter syllabus topics">
        <SyllabusSearch
          value={searchQuery}
          onChange={setSearchQuery}
          resultCount={filteredTopics.length}
          totalCount={syllabusTopics.length}
        />
        <CategoryFilter
          categories={categories}
          selectedCategory={selectedCategory}
          onChange={setSelectedCategory}
          categoryCounts={categoryCounts}
        />
      </section>

      {filteredTopics.length === 0 ? (
        <section className="syllabus-empty" aria-live="polite">
          <h2 className="syllabus-empty__title">No matching topics</h2>
          <p className="syllabus-empty__text">
            Try a different search term or category. You can also clear all filters to
            browse every syllabus topic.
          </p>
          <button type="button" className="syllabus-empty__button" onClick={clearFilters}>
            Clear filters
          </button>
        </section>
      ) : (
        <section
          className="syllabus-topics"
          aria-label="Syllabus topics"
          aria-live="polite"
        >
          {filteredTopics.map((topic) => (
            <SyllabusTopicCard key={topic.id} topic={topic} />
          ))}
        </section>
      )}
    </>
  );
}
