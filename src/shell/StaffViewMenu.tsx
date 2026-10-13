import { useEffect, useId, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Icon } from '../components/ui/Icon';
import { useSession } from '../context/SessionContext';
import { isAdmin } from '../context/session';
import { getRoleAreaFromPathname, staffViewPath, type StaffView } from '../lib/roleRoutes';

const VIEWS: { view: StaffView; label: string; hint: string }[] = [
  { view: 'admin', label: 'Admin', hint: 'Services, people, training and models' },
  { view: 'professor', label: 'Professor', hint: 'Courses as an instructor sees them' },
];

/**
 * Switch between the Admin and Professor interfaces.
 *
 * Production navigation for administrators, not a role switcher: the account
 * stays an administrator (the role beside the name says so), and choosing a
 * view only changes the URL. Professors have one interface and never see this.
 * The active view is read from the URL, so a refresh or a deep link shows the
 * right one with nothing remembered.
 *
 * A menu button: Enter, Space or ArrowDown opens it on the current view;
 * arrows, Home and End move; Escape closes it and returns focus here.
 */
export function StaffViewMenu() {
  const { session } = useSession();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const triggerId = useId();
  const menuId = useId();

  const area = getRoleAreaFromPathname(pathname);
  const current: StaffView | null =
    area === 'admin' || area === 'professor' ? area : area === null ? 'admin' : null;
  const currentLabel =
    area === 'student' ? 'Student preview' : VIEWS.find((v) => v.view === current)?.label;

  useEffect(() => {
    if (!open) {
      return;
    }
    const index = Math.max(
      0,
      VIEWS.findIndex((v) => v.view === current),
    );
    itemRefs.current[index]?.focus();

    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [open, current]);

  // A navigation from anywhere else (a link, Back) closes the menu.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  if (!isAdmin(session)) {
    return null;
  }

  function close() {
    setOpen(false);
    triggerRef.current?.focus();
  }

  function choose(view: StaffView) {
    close();
    if (view !== current) {
      navigate(staffViewPath(view, pathname));
    }
  }

  function onMenuKeyDown(event: React.KeyboardEvent) {
    const items = itemRefs.current.filter(Boolean) as HTMLButtonElement[];
    const at = items.indexOf(document.activeElement as HTMLButtonElement);
    const move = (index: number) => items[(index + items.length) % items.length]?.focus();
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        move(at + 1);
        break;
      case 'ArrowUp':
        event.preventDefault();
        move(at - 1);
        break;
      case 'Home':
        event.preventDefault();
        move(0);
        break;
      case 'End':
        event.preventDefault();
        move(items.length - 1);
        break;
      case 'Escape':
        event.preventDefault();
        close();
        break;
      case 'Tab':
        setOpen(false);
        break;
      default:
    }
  }

  return (
    <div className="shell-view" ref={rootRef}>
      <button
        ref={triggerRef}
        id={triggerId}
        type="button"
        className="ui-button ui-button--ghost ui-button--sm shell-view__trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => setOpen((value) => !value)}
        onKeyDown={(event) => {
          if (event.key === 'ArrowDown' && !open) {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        <span className="shell-view__caption">View</span>
        <span className="shell-view__current">{currentLabel}</span>
        <Icon name="expand" size={14} />
      </button>
      {open && (
        <ul
          id={menuId}
          role="menu"
          aria-labelledby={triggerId}
          className="shell-view__menu"
          onKeyDown={onMenuKeyDown}
        >
          {VIEWS.map((item, index) => {
            const checked = item.view === current;
            return (
              <li role="none" key={item.view}>
                <button
                  ref={(node) => {
                    itemRefs.current[index] = node;
                  }}
                  type="button"
                  role="menuitemradio"
                  aria-checked={checked}
                  tabIndex={-1}
                  className={
                    checked ? 'shell-view__item shell-view__item--current' : 'shell-view__item'
                  }
                  onClick={() => choose(item.view)}
                >
                  <span className="shell-view__item-label">{item.label}</span>
                  <span className="shell-view__item-hint">{item.hint}</span>
                  {checked && <Icon name="success" size={14} />}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
