import { EmptyState } from '../components/ui/EmptyState';
import { LinkButton } from '../components/ui/Button';

export function NotFoundPage() {
  return (
    <EmptyState
      size="full"
      title="Page not found"
      description="The page you are looking for doesn't exist or may have moved."
      action={
        <LinkButton to="/" variant="primary">
          Go back
        </LinkButton>
      }
    />
  );
}
