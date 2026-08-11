import { useCallback, useMemo, useState } from 'react';
import { useSessionContributions } from '../../hooks/useSessionContributions';
import { ContributeForm } from '../../components/contribute/ContributeForm';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { ConfirmDialog } from '../../components/ui/ConfirmDialog';
import { EmptyState } from '../../components/ui/EmptyState';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { useCourseId } from '../../context/CourseContext';
import { useSeedExamples } from '../../hooks/useSeedExamples';
import { toUserMessage } from '../../lib/errorMessages';
import type { SeedExample } from '../../types';

/**
 * Where students add example questions for their course.
 *
 * "Seed" never appears — it is our word for the record, not a thing a student
 * needs to know. Contributions are anonymous: no name is asked for and none is
 * stored.
 */
export function ContributePage() {
  const {
    seeds,
    loading,
    error,
    saving,
    saveError,
    addSeed,
    deleteSeed,
    clearSaveError,
  } = useSeedExamples();

  const { sessionIds, rememberSessionId, forgetSessionId } = useSessionContributions(
    useCourseId(),
  );
  const [justAdded, setJustAdded] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<SeedExample | null>(null);

  // Only student-written examples appear here. AI-generated ones live in the
  // instructor's review queue.
  const contributions = useMemo(
    () => seeds.filter((seed) => seed.origin === 'user'),
    [seeds],
  );

  /*
   * What this student added in this session.
   *
   * The full class list used to be shown here, which let anyone browse every
   * classmate's submitted answers — a privacy problem dressed up as a feature,
   * and not something a student needs. With no accounts there is no way to ask
   * "which of these are mine", and inventing a participant id to find out would
   * be worse. So we remember the ids created in this tab and show only those.
   * It is honest about its scope, it survives a reload, and it adds no identity
   * anywhere near the stored record.
   */
  const mine = useMemo(
    () => contributions.filter((seed) => sessionIds.includes(seed.id)),
    [contributions, sessionIds],
  );

  // Older records may be missing a section; skip those rather than offering
  // an empty suggestion.
  const sections = useMemo(() => {
    const unique = new Set<string>();
    for (const seed of seeds) {
      const section = seed.sourceSection?.trim();
      if (section && section !== 'Not specified') {
        unique.add(section);
      }
    }
    return [...unique].sort((left, right) => left.localeCompare(right));
  }, [seeds]);

  const handleSubmit = useCallback(
    async (example: SeedExample) => {
      clearSaveError();
      await addSeed(example);
      rememberSessionId(example.id);
      setJustAdded(true);
    },
    [addSeed, clearSaveError, rememberSessionId],
  );

  const confirmDelete = useCallback(async () => {
    if (!pendingDelete) {
      return;
    }
    clearSaveError();
    await deleteSeed(pendingDelete.id);
    forgetSessionId(pendingDelete.id);
    setPendingDelete(null);
  }, [clearSaveError, deleteSeed, forgetSessionId, pendingDelete]);

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        title="Contribute a Question"
        description="Add a question and the answer you would expect. Your instructor reviews contributions before they are used to improve the course assistant. Your name is not attached to anything you add."
      />

      {error && (
        <Callout tone="danger" title="Couldn't load your questions">
          {toUserMessage(new Error(error), {
            audience: 'student',
            context: 'examples-load',
          }).message}
        </Callout>
      )}

      {saveError && (
        <Callout tone="danger" title="Not saved">
          {toUserMessage(new Error(saveError), {
            audience: 'student',
            context: 'example-save',
          }).message}
        </Callout>
      )}

      {justAdded && !saveError && (
        <Callout tone="success" title="Added — thanks">
          Your question was added. Your instructor may review or edit it before
          it is used. Your name is not attached to it.
        </Callout>
      )}

      <ContributeForm
        existing={contributions}
        sections={sections}
        isSaving={saving}
        onSubmit={handleSubmit}
      />

      <section className="ui-stack">
        <SectionHeader
          title="Added in this session"
          description="Questions you added just now. Everyone's contributions go to your instructor for review."
          divider
        />

        {loading ? (
          <p className="ui-text-muted" role="status" aria-live="polite">
            Loading…
          </p>
        ) : mine.length === 0 ? (
          <EmptyState
            illustration="contribute"
            title="Nothing added yet"
            description="A good question is something you genuinely had to look up in the syllabus."
          />
        ) : (
          <ul className="contribution-list" aria-label="Questions you added">
            {mine.map((seed) => (
              <li key={seed.id} className="contribution">
                <p className="contribution__question">{seed.instruction}</p>
                <p className="contribution__answer">{seed.response}</p>
                <div className="contribution__footer">
                  {seed.sourceSection && seed.sourceSection !== 'Not specified' && (
                    <span className="contribution__section">{seed.sourceSection}</span>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    iconLeft="delete"
                    onClick={() => setPendingDelete(seed)}
                  >
                    Remove
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <ConfirmDialog
        open={pendingDelete !== null}
        tone="danger"
        title="Remove this question?"
        description="It will be deleted from this course. This cannot be undone."
        confirmLabel="Remove"
        onConfirm={() => void confirmDelete()}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
