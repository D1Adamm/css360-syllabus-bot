import { Link } from 'react-router-dom';
import { Navigation } from './Navigation';
import { PrototypeBanner } from './PrototypeBanner';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="app-layout">
      <PrototypeBanner />
      <header className="app-header">
        <div className="app-header__inner">
          <Link to="/" className="app-logo">
            Syllabus Model Lab
          </Link>
          <Navigation />
        </div>
      </header>
      <main className="app-main">{children}</main>
      <footer className="app-footer">
        <div className="app-footer__inner">
          <p>Syllabus Model Lab — Classroom prototype for model comparison research.</p>
        </div>
      </footer>
    </div>
  );
}
