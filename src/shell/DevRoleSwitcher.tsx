import { useId } from 'react';
import { useNavigate } from 'react-router-dom';
import { isRole, ROLES, type Role } from '../context/role';
import { useRole } from '../context/RoleContext';
import { roleHomePath } from '../lib/roleRoutes';

const ROLE_LABEL: Record<Role, string> = {
  student: 'Student',
  professor: 'Professor',
  admin: 'Admin',
};

/**
 * Development-only role switcher.
 *
 * Styled to look like scaffolding rather than a product feature, because that
 * is what it is: there is no authentication behind it and it grants nothing.
 * Every route stays reachable by URL whatever this says. When real sign-in
 * exists, delete this component and the `RoleProvider`'s stored value with it —
 * a real user never picks their own role.
 */
export interface DevRoleSwitcherProps {
  /**
   * The role area currently on screen, derived from the URL. Shown as the
   * selected value so the control can never contradict the visible chrome.
   */
  value: Role;
}

export function DevRoleSwitcher({ value }: DevRoleSwitcherProps) {
  const { setRole } = useRole();
  const navigate = useNavigate();
  const selectId = useId();

  function handleChange(value: string) {
    if (!isRole(value)) {
      return;
    }
    setRole(value);
    // Land on that role's home so the URL and the chrome never disagree.
    navigate(roleHomePath(value));
  }

  return (
    <div className="shell-devrole">
      <span className="shell-devrole__tag" aria-hidden="true">
        DEV
      </span>
      <label className="ui-visually-hidden" htmlFor={selectId}>
        Development role (not authentication)
      </label>
      <select
        id={selectId}
        className="shell-devrole__select"
        value={value}
        onChange={(event) => handleChange(event.target.value)}
        title="Development only — replaced when authentication is added"
      >
        {ROLES.map((value) => (
          <option key={value} value={value}>
            {ROLE_LABEL[value]}
          </option>
        ))}
      </select>
    </div>
  );
}
