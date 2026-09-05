import { StudentAccessPanel } from '../../components/invite/StudentAccessPanel';
import { PageHeader } from '../../components/ui/PageHeader';
import { formatCourseCode } from '../../lib/courseLabels';
import { useCourseId } from '../../context/CourseContext';
import { useCourseMetadata } from '../../hooks/useCourseMetadata';

/**
 * Inviting students: a join page and a class code.
 *
 * The backend issues a reusable classroom code bound to this course; every
 * student who enters it becomes a separate anonymous participant. This page
 * is reachable only for a course the professor holds a membership in, and the
 * backend checks that again on every action here.
 */
export function InviteStudentsPage() {
  const courseId = useCourseId();
  const { metadata } = useCourseMetadata(courseId);

  return (
    <div className="ui-stack ui-stack--section">
      <PageHeader
        eyebrow={formatCourseCode(metadata?.name)}
        title="Invite students"
        description="Put the join page and the class code on the board. Students enter the code on their laptop and are in."
      />
      <StudentAccessPanel courseId={courseId} courseName={metadata?.name} />
    </div>
  );
}
