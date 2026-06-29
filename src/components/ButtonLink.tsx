import { Link } from 'react-router-dom';

interface ButtonLinkProps {
  to: string;
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
}

export function ButtonLink({ to, children, variant = 'primary' }: ButtonLinkProps) {
  return (
    <Link
      to={to}
      className={`button-link button-link--${variant}`}
    >
      {children}
    </Link>
  );
}
