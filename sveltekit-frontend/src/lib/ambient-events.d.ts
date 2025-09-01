// Temporary ambient event declarations to avoid 'parameter of type never' diagnostics
// during migration. Remove after explicit dispatch typings are added.

// Broad augmentation (kept minimal) to cover common handlers.
// This intentionally widens types; narrowing will follow once components export events.

declare namespace svelteHTML {
  interface HTMLAttributes<T> {
    onclick?: (event: MouseEvent) => any;
    ondblclick?: (event: MouseEvent) => any;
    onkeydown?: (event: KeyboardEvent) => any;
    onkeyup?: (event: KeyboardEvent) => any;
    onkeypress?: (event: KeyboardEvent) => any;
    onchange?: (event: Event) => any;
    oninput?: (event: InputEvent | Event) => any;
    onsubmit?: (event: Event) => any;
    onfocus?: (event: FocusEvent) => any;
    onblur?: (event: FocusEvent) => any;
  }
}


