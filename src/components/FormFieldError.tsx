interface FormFieldErrorProps {
  id: string;
  message: string;
}

export function FormFieldError({ id, message }: FormFieldErrorProps) {
  return (
    <p id={id} className="form-field-error" role="alert">
      {message}
    </p>
  );
}
