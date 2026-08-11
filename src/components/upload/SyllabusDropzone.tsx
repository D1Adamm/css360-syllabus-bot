import { useId, useRef, useState } from 'react';
import { Icon } from '../ui/Icon';

export interface SyllabusDropzoneProps {
  file: File | null;
  onSelect: (file: File | null) => void;
  disabled?: boolean;
  error?: string;
  errorId?: string;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * File selection for the syllabus.
 *
 * Drag-and-drop with a real, focusable file input underneath — the input is
 * the control, the drop area is an affordance on top of it. That keeps
 * keyboard and screen-reader users on the standard path instead of a
 * div-pretending-to-be-a-button.
 *
 * Validation stays with the form; this component only reports the selection.
 */
export function SyllabusDropzone({
  file,
  onSelect,
  disabled = false,
  error,
  errorId,
}: SyllabusDropzoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleDrop(event: React.DragEvent) {
    event.preventDefault();
    setDragging(false);
    if (disabled) {
      return;
    }
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      onSelect(dropped);
    }
  }

  return (
    <div className="dropzone-field">
      <label className="ui-field__label" htmlFor={inputId}>
        Syllabus file <span className="ui-field__requirement">(required)</span>
      </label>

      <div
        className={[
          'dropzone',
          dragging ? 'dropzone--dragging' : '',
          file ? 'dropzone--selected' : '',
          error ? 'dropzone--invalid' : '',
          disabled ? 'dropzone--disabled' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) {
            setDragging(true);
          }
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          id={inputId}
          className="dropzone__input"
          type="file"
          accept=".pdf,.txt,application/pdf,text/plain"
          disabled={disabled}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errorId : `${inputId}-hint`}
          onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
        />

        {file ? (
          <div className="dropzone__selected">
            <Icon name="syllabus" size={20} />
            <span className="dropzone__filename">{file.name}</span>
            <span className="dropzone__filesize">{formatSize(file.size)}</span>
            <button
              type="button"
              className="dropzone__clear"
              disabled={disabled}
              onClick={() => {
                onSelect(null);
                if (inputRef.current) {
                  inputRef.current.value = '';
                }
              }}
            >
              Choose a different file
            </button>
          </div>
        ) : (
          <div className="dropzone__prompt">
            <Icon name="upload" size={22} />
            <p className="dropzone__title">Drop your syllabus here, or browse</p>
            <p className="dropzone__hint" id={`${inputId}-hint`}>
              PDF or plain text
            </p>
          </div>
        )}
      </div>

      {error && (
        <p className="ui-field__error" id={errorId} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
