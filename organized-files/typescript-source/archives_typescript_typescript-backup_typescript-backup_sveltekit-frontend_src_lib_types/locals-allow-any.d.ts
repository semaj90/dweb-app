// Lightweight ambient augmentation to reduce noise from missing Locals properties
// This intentionally allows arbitrary properties on App.Locals during incremental remediation.
declare namespace App {
  interface Locals {
    [key: string]: any;
  }
}
