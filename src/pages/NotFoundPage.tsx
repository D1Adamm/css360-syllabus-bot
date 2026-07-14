import { ButtonLink } from '../components/ButtonLink';
import { DEFAULT_COURSE_ID } from '../lib/courseId';
import { coursePagePath } from '../lib/courseRoutes';

export function NotFoundPage() {
  return (
    <section className="not-found" aria-labelledby="not-found-title">
      <h1 id="not-found-title" className="not-found__title">
        Page Not Found
      </h1>
      <p className="not-found__text">
        The page you are looking for does not exist or may have been moved.
      </p>
      <ButtonLink to={coursePagePath(DEFAULT_COURSE_ID, 'home')}>Return to Home</ButtonLink>
    </section>
  );
}
