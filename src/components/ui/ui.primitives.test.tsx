/** @vitest-environment jsdom */
import { useState } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { Illustration } from '../illustration/Illustration';
import { Button, LinkButton } from './Button';
import { Callout } from './Callout';
import { ConfirmDialog } from './ConfirmDialog';
import { EmptyState } from './EmptyState';
import { FormField } from './FormField';
import { Icon } from './Icon';
import { ICON_NAMES } from './icons';
import { ProgressSteps } from './ProgressSteps';
import { StatusPill } from './StatusPill';
import { Surface } from './Surface';

afterEach(() => {
  cleanup();
});

describe('Button', () => {
  it('defaults to type="button" so it never submits a form by accident', () => {
    render(<Button>Save</Button>);
    expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute(
      'type',
      'button',
    );
  });

  it('applies the variant and size classes that carry the action hierarchy', () => {
    render(
      <Button variant="danger" size="sm">
        Delete
      </Button>,
    );
    const button = screen.getByRole('button', { name: 'Delete' });
    expect(button).toHaveClass('ui-button--danger');
    expect(button).toHaveClass('ui-button--sm');
  });

  it('blocks activation and announces busy while loading', () => {
    const onClick = vi.fn();
    render(
      <Button loading loadingLabel="Preparing…" onClick={onClick}>
        Request model
      </Button>,
    );

    const button = screen.getByRole('button', { name: 'Preparing…' });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('aria-busy', 'true');

    fireEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });

  it('forwards a ref to the underlying button element', () => {
    const ref = { current: null as HTMLButtonElement | null };
    render(<Button ref={ref}>Focus me</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });

  it('renders LinkButton as a link, not a button', () => {
    render(
      <MemoryRouter>
        <LinkButton to="/somewhere" variant="primary">
          Go
        </LinkButton>
      </MemoryRouter>,
    );
    expect(screen.getByRole('link', { name: 'Go' })).toHaveAttribute(
      'href',
      '/somewhere',
    );
  });
});

describe('Callout', () => {
  it('announces danger assertively as an alert', () => {
    render(
      <Callout tone="danger" title="Unavailable">
        Try again in a moment.
      </Callout>,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Unavailable');
  });

  it('announces other tones politely as a status', () => {
    render(<Callout tone="success">Syllabus ready</Callout>);
    const status = screen.getByRole('status');
    expect(status).toHaveAttribute('aria-live', 'polite');
    expect(status).toHaveTextContent('Syllabus ready');
  });

  it('can opt out of live-region announcement entirely', () => {
    render(<Callout live={false}>Static note</Callout>);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('StatusPill', () => {
  it('renders its tone class and label', () => {
    const { container } = render(<StatusPill tone="accent">Ready</StatusPill>);
    expect(container.querySelector('.ui-pill--accent')).toBeInTheDocument();
    expect(screen.getByText('Ready')).toBeInTheDocument();
  });

  it('omits the decorative dot when asked', () => {
    const { container } = render(<StatusPill dot={false}>AI Generated</StatusPill>);
    expect(container.querySelector('.ui-pill__dot')).not.toBeInTheDocument();
  });
});

describe('FormField', () => {
  it('associates the label with the control it renders', () => {
    render(
      <FormField label="Your question">
        {({ id }) => <textarea id={id} />}
      </FormField>,
    );
    expect(screen.getByLabelText('Your question')).toBeInTheDocument();
  });

  it('wires hint and error into aria-describedby and marks the field invalid', () => {
    render(
      <FormField label="Expected answer" hint="Keep it factual." error="Required.">
        {({ id, describedBy, invalid }) => (
          <input id={id} aria-describedby={describedBy} aria-invalid={invalid} />
        )}
      </FormField>,
    );

    const input = screen.getByLabelText(/Expected answer/);
    expect(input).toHaveAttribute('aria-invalid', 'true');

    const describedBy = input.getAttribute('aria-describedby') ?? '';
    const describedIds = describedBy.split(' ').filter(Boolean);
    expect(describedIds).toHaveLength(2);

    const describedText = describedIds
      .map((id) => document.getElementById(id)?.textContent)
      .join(' ');
    expect(describedText).toContain('Keep it factual.');
    expect(describedText).toContain('Required.');
  });

  it('surfaces the error message to assistive technology', () => {
    render(
      <FormField label="Term" error="Term is required.">
        {({ id }) => <input id={id} />}
      </FormField>,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Term is required.');
  });
});

describe('ProgressSteps', () => {
  it('marks the active step with aria-current and completes earlier steps', () => {
    render(
      <ProgressSteps
        currentIndex={1}
        steps={[
          { id: 'contribute', label: 'Contribute' },
          { id: 'compare', label: 'Compare' },
          { id: 'evaluate', label: 'Evaluate' },
        ]}
      />,
    );

    const items = screen.getAllByRole('listitem');
    expect(items[0]).toHaveClass('ui-steps__item--complete');
    expect(items[1]).toHaveAttribute('aria-current', 'step');
    expect(items[2]).toHaveClass('ui-steps__item--upcoming');
  });

  it('treats every step as complete when the sequence is finished', () => {
    render(
      <ProgressSteps
        currentIndex={2}
        allComplete
        steps={[
          { id: 'a', label: 'A' },
          { id: 'b', label: 'B' },
        ]}
      />,
    );

    for (const item of screen.getAllByRole('listitem')) {
      expect(item).toHaveClass('ui-steps__item--complete');
    }
  });
});

describe('ConfirmDialog', () => {
  const baseProps = {
    title: 'Delete all evaluation data?',
    description: 'This cannot be undone.',
    confirmLabel: 'Delete',
    cancelLabel: 'Cancel',
  };

  it('renders nothing while closed', () => {
    render(
      <ConfirmDialog {...baseProps} open={false} onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
  });

  it('is a labelled modal that starts focus on the safe action', () => {
    render(
      <ConfirmDialog {...baseProps} open onConfirm={vi.fn()} onCancel={vi.fn()} />,
    );

    const dialog = screen.getByRole('alertdialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAccessibleName('Delete all evaluation data?');
    expect(dialog).toHaveAccessibleDescription('This cannot be undone.');
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();
  });

  it('cancels on Escape', () => {
    const onCancel = vi.fn();
    render(<ConfirmDialog {...baseProps} open onConfirm={vi.fn()} onCancel={onCancel} />);

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('runs the confirm and cancel callbacks from their buttons', () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(
      <ConfirmDialog {...baseProps} open onConfirm={onConfirm} onCancel={onCancel} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(onConfirm).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('restores focus to the trigger when it closes', () => {
    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open
          </button>
          <ConfirmDialog
            {...baseProps}
            open={open}
            onConfirm={() => setOpen(false)}
            onCancel={() => setOpen(false)}
          />
        </>
      );
    }

    render(<Harness />);
    const trigger = screen.getByRole('button', { name: 'Open' });
    trigger.focus();
    fireEvent.click(trigger);

    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(trigger).toHaveFocus();
  });
});

describe('EmptyState and Illustration', () => {
  it('renders a placeholder when no illustration asset exists yet', () => {
    const { container } = render(<Illustration name="contribute" />);
    expect(container.querySelector('.ui-illustration--placeholder')).toBeInTheDocument();
    expect(container.querySelector('img')).not.toBeInTheDocument();
  });

  it('hides decorative illustrations from assistive technology', () => {
    const { container } = render(<Illustration name="landing" />);
    expect(container.querySelector('.ui-illustration')).toHaveAttribute(
      'aria-hidden',
      'true',
    );
  });

  it('exposes an accessible name when an illustration is not decorative', () => {
    render(<Illustration name="model-ready" decorative={false} alt="Model is ready" />);
    expect(screen.getByRole('img', { name: 'Model is ready' })).toBeInTheDocument();
  });

  it('renders title, description and action together', () => {
    render(
      <EmptyState
        illustration="contribute"
        title="No questions yet"
        description="Contribute the first one."
        action={<Button variant="primary">Contribute</Button>}
      />,
    );

    expect(screen.getByText('No questions yet')).toBeInTheDocument();
    expect(screen.getByText('Contribute the first one.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Contribute' })).toBeInTheDocument();
  });
});

describe('Surface', () => {
  it('is plain and unbordered by default so it does not become a card', () => {
    const { container } = render(<Surface>content</Surface>);
    const surface = container.querySelector('.ui-surface');
    expect(surface).toHaveClass('ui-surface--plain');
    expect(surface).not.toHaveClass('ui-surface--bordered');
  });

  it('renders the requested element type', () => {
    const { container } = render(
      <Surface as="section" tone="raised" padding="md">
        content
      </Surface>,
    );
    expect(container.querySelector('section.ui-surface--raised')).toBeInTheDocument();
  });
});

describe('Icon', () => {
  it('hides decorative icons from assistive technology', () => {
    const { container } = render(<Icon name="syllabus" />);
    expect(container.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
  });

  it('exposes a labelled icon as an image', () => {
    render(<Icon name="health" label="System health" />);
    expect(screen.getByRole('img', { name: 'System health' })).toBeInTheDocument();
  });

  it('resolves every registered icon name', () => {
    expect(ICON_NAMES.length).toBeGreaterThan(0);
    for (const name of ICON_NAMES) {
      const { container, unmount } = render(<Icon name={name} />);
      expect(container.querySelector('svg')).toBeInTheDocument();
      unmount();
    }
  });
});
