interface FormFieldErrorProps {
  id: string;
  message: string;
}

export function FormFieldError({ id, message }: FormFieldErrorProps) {
  return (
    <p id={id} className="ui-field__error" role="alert">
      {message}
    </p>
  );
}
