import { useState } from 'react';
import { Button } from '../../components/ui/Button';
import { Callout } from '../../components/ui/Callout';
import { Icon } from '../../components/ui/Icon';
import { formatCourseCode } from '../../lib/courseLabels';
import { PageHeader } from '../../components/ui/PageHeader';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { useCourseId } from '../../context/CourseContext';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';
import { studentCourseHomePath } from '../../lib/roleRoutes';

/**
 * Inviting students — an integration boundary, not a feature.
 *
 * There is no enrolment backend: no membership records, no join codes, no
 * authentication. Rather than mint a code that grants nothing and store it
 * somewhere that implies it means something, this page shows the real link a
 * student can already use and states plainly that access is not yet
 * restricted.
 *
 * The join-code layout below is the shape this page takes once the backend can
 * issue and redeem codes. See `docs/frontend-backend-gaps.md`.
 */
export function InviteStudentsPage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);
  const [copied, setCopied] = useState(false);

  const path = studentCourseHomePath(courseId);
  const link =
    typeof window === 'undefined' ? path : `${window.location.origin}${path}`;

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2500);
    } catch {
      // Clipboard access can be refused; the link is selectable either way.
      setCopied(false);
    }
  }

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        eyebrow={formatCourseCode(metadata?.name)}
        title="Invite students"
        description="Share this link so students can open the course and start contributing questions."
      />

      <section className="ui-stack ui-stack--snug">
        <SectionHeader title="Course link" level={2} divider />
        <div className="invite__link-row">
          <code className="invite__link">{link}</code>
          <Button
            variant="secondary"
            iconLeft="copy"
            onClick={() => void copyLink()}
          >
            {copied ? 'Copied' : 'Copy link'}
          </Button>
        </div>
        <p className="ui-text-xs ui-text-muted">
          Anyone with this link can open the course. Sign-in and class rosters
          are not built yet, so treat the link as you would a shared document.
        </p>
      </section>

      <Callout tone="warning" title="Join codes are not available yet">
        This project does not yet have student accounts or course membership, so
        there is nothing to issue a join code against. When that exists, a short
        code and a QR code will appear here alongside the link.
      </Callout>

      <section className="ui-stack ui-stack--snug">
        <SectionHeader
          title="What this will look like"
          description="Shown so the layout is settled before the backend work starts."
          level={2}
          divider
        />
        <div className="invite__preview" aria-label="Preview of the future join code panel">
          <div className="invite__preview-code" aria-hidden="true">
            <span className="invite__preview-label">Join code</span>
            <span className="invite__preview-value">— — — — — —</span>
          </div>
          <div className="invite__preview-qr" aria-hidden="true">
            <Icon name="link" size={28} />
            <span>QR code</span>
          </div>
          <p className="invite__preview-note">
            Not functional. Nothing here is generated or stored.
          </p>
        </div>
      </section>
    </div>
  );
}
