import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

/*
 * Stylesheet order: tokens, base, then components and page styles.
 *
 * The legacy `global.css` is gone — every page now uses the design system, so
 * there is nothing left for it to style.
 */
import './styles/fonts.css';
import './styles/tokens.css';
import './styles/base.css';
import './components/ui/ui.css';
import './styles/patterns.css';
import './styles/student.css';
import './styles/professor.css';
import './styles/admin.css';
import './styles/utilities.css';

import App from './App.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
