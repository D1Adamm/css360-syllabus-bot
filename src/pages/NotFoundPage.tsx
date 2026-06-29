import { ButtonLink } from '../components/ButtonLink';

export function NotFoundPage() {
  return (
    <section className="not-found" aria-labelledby="not-found-title">
      <h1 id="not-found-title" className="not-found__title">
        Page Not Found
      </h1>
      <p className="not-found__text">
        The page you are looking for does not exist or may have been moved.
      </p>
      <ButtonLink to="/">Return to Home</ButtonLink>
    </section>
  );
}
