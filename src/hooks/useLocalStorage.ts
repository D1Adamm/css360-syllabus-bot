import { useCallback, useEffect, useState } from 'react';

function readStoredValue<T>(
  key: string,
  defaultValue: T,
  validate?: (value: unknown) => value is T,
): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) {
      return defaultValue;
    }

    const parsed: unknown = JSON.parse(raw);
    if (validate && !validate(parsed)) {
      return defaultValue;
    }

    return parsed as T;
  } catch {
    return defaultValue;
  }
}

export function useLocalStorage<T>(
  key: string,
  defaultValue: T,
  validate?: (value: unknown) => value is T,
) {
  const [value, setValue] = useState<T>(() =>
    readStoredValue(key, defaultValue, validate),
  );

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Ignore quota or serialization errors to keep the UI responsive.
    }
  }, [key, value]);

  const reset = useCallback(() => {
    setValue(defaultValue);
    try {
      localStorage.removeItem(key);
    } catch {
      // Ignore storage errors during reset.
    }
  }, [defaultValue, key]);

  return [value, setValue, reset] as const;
}
