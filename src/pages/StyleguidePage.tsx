import { useState } from 'react';
import './styleguide.css';
import { ILLUSTRATION_NAMES } from '../assets/illustrations';
import { Illustration } from '../components/illustration/Illustration';
import { Button, LinkButton } from '../components/ui/Button';
import { Callout } from '../components/ui/Callout';
import { ConfirmDialog } from '../components/ui/ConfirmDialog';
import { EmptyState } from '../components/ui/EmptyState';
import { FormField } from '../components/ui/FormField';
import { Icon } from '../components/ui/Icon';
import { ICON_NAMES } from '../components/ui/icons';
import { PageHeader } from '../components/ui/PageHeader';
import { ProgressSteps } from '../components/ui/ProgressSteps';
import { SectionHeader } from '../components/ui/SectionHeader';
import { StatusPill } from '../components/ui/StatusPill';
import { Surface } from '../components/ui/Surface';

/**
 * TEMPORARY — Phase 1 review surface.
 *
 * This page exists so the design system can be inspected before any feature
 * page is redesigned. It is not part of the product and is removed in Phase 10.
 */

const SWATCHES: { group: string; tokens: string[] }[] = [
  {
    group: 'Purple — brand',
    tokens: [
      '--purple-900',
      '--purple-800',
      '--purple-700',
      '--purple-600',
      '--purple-400',
      '--purple-200',
      '--purple-100',
      '--purple-50',
    ],
  },
  {
    group: 'Gold — restrained accent',
    tokens: ['--gold-700', '--gold-500', '--gold-300', '--gold-100'],
  },
  {
    group: 'Neutral',
    tokens: [
      '--ink-900',
      '--ink-700',
      '--ink-500',
      '--ink-400',
      '--line-300',
      '--line-200',
      '--canvas-sunken',
      '--canvas',
      '--surface',
    ],
  },
  {
    group: 'Semantic',
    tokens: [
      '--success-700',
      '--success-100',
      '--warning-700',
      '--warning-100',
      '--danger-700',
      '--danger-100',
    ],
  },
];

const TYPE_SCALE: { token: string; label: string; display?: boolean }[] = [
  { token: '--text-3xl', label: 'Page title — Source Serif 4', display: true },
  { token: '--text-2xl', label: 'Course title — Source Serif 4', display: true },
  { token: '--text-xl', label: 'Section heading — Inter' },
  { token: '--text-lg', label: 'Subsection heading — Inter' },
  { token: '--text-base', label: 'Body — Inter' },
  { token: '--text-sm', label: 'Secondary body — Inter' },
  { token: '--text-xs', label: 'Meta and labels — Inter' },
  { token: '--text-2xs', label: 'Smallest label — Inter' },
];

const SPACE_TOKENS = [
  '--space-1',
  '--space-2',
  '--space-3',
  '--space-4',
  '--space-6',
  '--space-8',
  '--space-12',
  '--space-16',
];

export function StyleguidePage() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dangerDialogOpen, setDangerDialogOpen] = useState(false);
  const [question, setQuestion] = useState('');

  return (
    <div className="ui-root sg-page">
      <div className="ui-container ui-stack ui-stack--section">
        <PageHeader
          eyebrow="Phase 1 · temporary review surface"
          title="Design system"
          description="Tokens, typography and UI primitives for the Syllabus Model Lab redesign. No feature page uses these yet."
          actions={<Button variant="secondary" iconLeft="settings">Placeholder action</Button>}
        />

        {/* ---------------------------------------------------------- colour */}
        <section className="ui-stack">
          <SectionHeader
            title="Colour"
            description="Purple carries the brand. Gold is an accent for ready states, completion and active navigation — never a large fill."
            divider
          />
          <div className="ui-stack ui-stack--loose">
            {SWATCHES.map((group) => (
              <div key={group.group} className="ui-stack ui-stack--tight">
                <p className="ui-text-sm ui-text-muted">{group.group}</p>
                <div className="sg-swatches">
                  {group.tokens.map((token) => (
                    <div key={token} className="sg-swatch">
                      <span
                        className="sg-swatch__chip"
                        style={{ backgroundColor: `var(${token})` }}
                      />
                      <code className="sg-swatch__name">{token}</code>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ----------------------------------------------------- typography */}
        <section className="ui-stack">
          <SectionHeader
            title="Typography"
            description="Inter for all UI. Source Serif 4 only for page and course titles."
            divider
          />
          <div className="ui-stack ui-stack--snug">
            {TYPE_SCALE.map((row) => (
              <div key={row.token} className="sg-type-row">
                <span
                  className="sg-type-row__sample"
                  style={{
                    fontSize: `var(${row.token})`,
                    fontFamily: row.display ? 'var(--font-display)' : 'var(--font-sans)',
                    fontWeight: row.display ? 600 : 400,
                  }}
                >
                  {row.label}
                </span>
                <code className="ui-text-xs ui-text-muted">{row.token}</code>
              </div>
            ))}
          </div>
        </section>

        {/* --------------------------------------------------------- spacing */}
        <section className="ui-stack">
          <SectionHeader
            title="Spacing"
            description="A 4px base scale. Sections are separated by --rhythm-section (3rem)."
            divider
          />
          <div className="ui-stack ui-stack--tight">
            {SPACE_TOKENS.map((token) => (
              <div key={token} className="sg-space-row">
                <span className="sg-space-bar" style={{ width: `var(${token})` }} />
                <code className="ui-text-xs ui-text-muted">{token}</code>
              </div>
            ))}
          </div>
        </section>

        {/* --------------------------------------------------------- buttons */}
        <section className="ui-stack">
          <SectionHeader
            title="Buttons"
            description="One primary action per view. Everything else steps down the hierarchy."
            divider
          />
          <div className="ui-row">
            <Button variant="primary">Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="tertiary">Tertiary</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="danger">Delete everything</Button>
          </div>
          <div className="ui-row">
            <Button variant="primary" iconRight="forward">
              With trailing icon
            </Button>
            <Button variant="secondary" iconLeft="upload">
              With leading icon
            </Button>
            <Button variant="primary" loading loadingLabel="Preparing…">
              Preparing
            </Button>
            <Button variant="primary" disabled>
              Disabled
            </Button>
            <Button variant="secondary" size="sm">
              Small
            </Button>
            <LinkButton to="/styleguide" variant="tertiary" iconRight="next">
              Router link button
            </LinkButton>
          </div>
        </section>

        {/* -------------------------------------------------------- surfaces */}
        <section className="ui-stack">
          <SectionHeader
            title="Surfaces"
            description="Plain and unbordered by default. Not everything needs to be a card."
            divider
          />
          <div className="ui-grid">
            <Surface tone="plain" padding="md" bordered>
              <p className="ui-text-sm">Plain with a hairline border</p>
            </Surface>
            <Surface tone="raised" padding="md">
              <p className="ui-text-sm">Raised — a discrete object</p>
            </Surface>
            <Surface tone="sunken" padding="md">
              <p className="ui-text-sm">Sunken — a recessed area</p>
            </Surface>
            <Surface tone="accent" padding="md">
              <p className="ui-text-sm">Accent — sparing purple tint</p>
            </Surface>
          </div>
        </section>

        {/* -------------------------------------------------------- callouts */}
        <section className="ui-stack">
          <SectionHeader
            title="Callouts"
            description="Replaces the seven bespoke notice styles in the legacy stylesheet."
            divider
          />
          <div className="ui-stack ui-stack--snug">
            <Callout tone="info" title="Your example was submitted">
              A course instructor reviews new questions before they are used to
              improve the course assistant.
            </Callout>
            <Callout tone="success" title="Syllabus ready">
              Your syllabus has been processed and is ready for students.
            </Callout>
            <Callout tone="warning" title="12 examples need review">
              These are waiting for you before the course model can be requested.
            </Callout>
            <Callout
              tone="danger"
              title="This response is temporarily unavailable"
              actions={
                <Button size="sm" variant="secondary">
                  Try again
                </Button>
              }
            >
              Try again in a moment.
            </Callout>
          </div>
        </section>

        {/* ----------------------------------------------------------- pills */}
        <section className="ui-stack">
          <SectionHeader
            title="Status pills"
            description="One per object. Gold is reserved for ready states."
            divider
          />
          <div className="ui-row">
            <StatusPill tone="neutral">Not requested</StatusPill>
            <StatusPill tone="info">Requested</StatusPill>
            <StatusPill tone="progress">Preparing</StatusPill>
            <StatusPill tone="accent">Ready</StatusPill>
            <StatusPill tone="success">Approved</StatusPill>
            <StatusPill tone="warning">Needs review</StatusPill>
            <StatusPill tone="danger">Failed</StatusPill>
            <StatusPill tone="neutral" dot={false}>
              AI Generated
            </StatusPill>
          </div>
        </section>

        {/* -------------------------------------------------------- progress */}
        <section className="ui-stack">
          <SectionHeader
            title="Progress steps"
            description="The student spine: Contribute → Compare → Evaluate."
            divider
          />
          <ProgressSteps
            currentIndex={1}
            steps={[
              { id: 'contribute', label: 'Contribute', meta: '3 questions' },
              { id: 'compare', label: 'Compare', meta: '2 completed' },
              { id: 'evaluate', label: 'Evaluate', meta: 'Not started' },
            ]}
          />
        </section>

        {/* ----------------------------------------------------------- forms */}
        <section className="ui-stack">
          <SectionHeader
            title="Form fields"
            description="Label, hint, control and error as one accessible unit."
            divider
          />
          <div className="sg-form ui-stack ui-stack--loose">
            <FormField
              label="Your question"
              hint="Think of something another student might reasonably ask about this course."
              required
            >
              {({ id, describedBy, invalid }) => (
                <textarea
                  id={id}
                  aria-describedby={describedBy}
                  aria-invalid={invalid}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="How much of my grade comes from the final project?"
                />
              )}
            </FormField>

            <FormField
              label="Expected answer"
              error="An expected answer is required."
              required
            >
              {({ id, describedBy, invalid }) => (
                <input id={id} aria-describedby={describedBy} aria-invalid={invalid} />
              )}
            </FormField>

            <FormField label="Relevant syllabus section" optional>
              {({ id, describedBy }) => (
                <select id={id} aria-describedby={describedBy} defaultValue="">
                  <option value="">Choose a section…</option>
                  <option value="grading">Grading &amp; Assessment</option>
                  <option value="policies">Course Policies</option>
                </select>
              )}
            </FormField>
          </div>
        </section>

        {/* --------------------------------------------------------- dialogs */}
        <section className="ui-stack">
          <SectionHeader
            title="Confirmation dialog"
            description="Replaces every window.confirm. Focus starts on Cancel; Escape and the backdrop both cancel."
            divider
          />
          <div className="ui-row">
            <Button variant="secondary" onClick={() => setDialogOpen(true)}>
              Open dialog
            </Button>
            <Button variant="danger" onClick={() => setDangerDialogOpen(true)}>
              Open destructive dialog
            </Button>
          </div>

          <ConfirmDialog
            open={dialogOpen}
            title="Request the course model?"
            description="Your 48 approved examples will be used to prepare a model for this course. You can keep reviewing examples while it is prepared."
            confirmLabel="Request model"
            onConfirm={() => setDialogOpen(false)}
            onCancel={() => setDialogOpen(false)}
          />

          <ConfirmDialog
            open={dangerDialogOpen}
            tone="danger"
            title="Delete all evaluation data?"
            description="Every evaluation submitted for this course will be permanently removed. This cannot be undone."
            confirmLabel="Delete evaluations"
            onConfirm={() => setDangerDialogOpen(false)}
            onCancel={() => setDangerDialogOpen(false)}
          />
        </section>

        {/* ---------------------------------------------------- empty states */}
        <section className="ui-stack">
          <SectionHeader
            title="Empty states"
            description="Illustration slots fall back to a placeholder until custom artwork is added."
            divider
          />
          <Surface tone="raised" padding="md">
            <EmptyState
              illustration="contribute"
              title="No questions yet"
              description="Contribute the first question for this course and see how different AI approaches answer it."
              action={
                <Button variant="primary" iconLeft="contribute">
                  Contribute a question
                </Button>
              }
            />
          </Surface>
        </section>

        {/* --------------------------------------------------- illustrations */}
        <section className="ui-stack">
          <SectionHeader
            title="Illustration slots"
            description="Four named slots, all currently unfilled. Adding an SVG changes nothing else."
            divider
          />
          <div className="ui-grid">
            {ILLUSTRATION_NAMES.map((name) => (
              <Surface key={name} tone="plain" padding="md" bordered>
                <div className="ui-stack ui-stack--tight sg-illustration">
                  <Illustration name={name} size="md" />
                  <code className="ui-text-xs ui-text-muted">{name}</code>
                </div>
              </Surface>
            ))}
          </div>
        </section>

        {/* ----------------------------------------------------------- icons */}
        <section className="ui-stack">
          <SectionHeader
            title="Icons"
            description="Requested by meaning, never by vendor name. lucide-react sits behind Icon.tsx."
            divider
          />
          <div className="sg-icons">
            {ICON_NAMES.map((name) => (
              <div key={name} className="sg-icon">
                <Icon name={name} size={20} />
                <code className="ui-text-xs ui-text-muted">{name}</code>
              </div>
            ))}
          </div>
        </section>

        <p className="ui-text-xs ui-text-muted sg-footer">
          Temporary Phase 1 review page. Removed in Phase 10.
        </p>
      </div>
    </div>
  );
}
