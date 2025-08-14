<script lang="ts">
  import Button from "$lib/components/ui/Button.svelte";
  // FIX: Import all necessary parts of the Tooltip component
  import Card from "$lib/components/ui/Card.svelte";
  import Form from "$lib/components/ui/Form.svelte";
  import Input from "$lib/components/ui/Input.svelte";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import TooltipContent from "$lib/components/ui/TooltipContent.svelte";
  import TooltipTrigger from "$lib/components/ui/TooltipTrigger.svelte";
  import { modals } from "$lib/stores/modal";
  import { notifications } from "$lib/stores/notification";

  // Demo notification functions
  function showSuccessNotification() {
    notifications.success(
      "Success!",
      "Your action was completed successfully.",
      {
        actions: [
          {
            label: "View Details",
            action: () => console.log("View details clicked"),
            variant: "primary",
          },
        ],
}
    );
}
  function showErrorNotification() {
    notifications.error(
      "Error Occurred",
      "Something went wrong. Please try again.",
      {
        duration: 0, // Don't auto-dismiss
        actions: [
          {
            label: "Retry",
            action: () => console.log("Retry clicked"),
            variant: "primary",
          },
          {
            label: "Cancel",
            action: () => console.log("Cancel clicked"),
            variant: "secondary",
          },
        ],
}
    );
}
  function showWarningNotification() {
    notifications.warning(
      "Warning",
      "This action may have consequences. Please review before proceeding."
    );
}
  function showInfoNotification() {
    notifications.info(
      "Information",
      "Here is some useful information about the current process."
    );
}
  // Demo modal functions
  function showConfirmModal() {
    modals.confirm({
      title: "Delete Case",
      message:
        "Are you sure you want to delete this case? This action cannot be undone.",
      confirmText: "Delete",
      cancelText: "Cancel",
      onConfirm: () => {
        notifications.success(
          "Case deleted",
          "The case has been successfully deleted."
        );
      },
      onCancel: () => {
        notifications.info("Cancelled", "Delete operation was cancelled.");
      },
    });
}
  function showAlertModal() {
    modals.alert({
      title: "System Maintenance",
      message:
        "The system will be undergoing maintenance from 2:00 AM to 4:00 AM EST.",
      onClose: () => {
        console.log("Alert dismissed");
      },
    });
}
  function showPromptModal() {
    modals.prompt({
      title: "Add Note",
      message: "Please enter a note for this case:",
      placeholder: "Enter your note here...",
      defaultValue: "",
      onConfirm: (value) => {
        notifications.success(
          "Note added",
          `Note: "${value}" has been added to the case.`
        );
      },
      onCancel: () => {
        console.log("Prompt cancelled");
      },
    });
}
  // Demo form
  const demoFormOptions = {
    initialValues: {
      email: "",
      password: "",
      confirmPassword: "",
      name: "",
      terms: false,
    },
    validators: {
      email: (value: string) => {
        if (!value) return "Email is required";
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value))
          return "Invalid email format";
        return null;
      },
      password: (value: string) => {
        if (!value) return "Password is required";
        if (value.length < 8) return "Password must be at least 8 characters";
        return null;
      },
      confirmPassword: (value: string) => {
        if (!value) return "Please confirm your password";
        // Note: For now we can't cross-validate with password field
        return null;
      },
      name: (value: string) => {
        if (!value) return "Name is required";
        if (value.length < 2) return "Name must be at least 2 characters";
        return null;
      },
      terms: (value: boolean) => {
        if (!value) return "You must accept the terms and conditions";
        return null;
      },
    },
    requiredFields: ["email", "password", "confirmPassword", "name", "terms"],
  };

  function handleDemoFormSubmit(event: CustomEvent) {
    const { values, isValid } = event.detail;
    if (isValid) {
      notifications.success(
        "Form submitted!",
        `Welcome ${values.name}! Your account has been created.`
      );
}}
  let interactiveCardSelected = false;
</script>

<svelte:head>
  <title>UI Components Demo - Legal Case Management</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <!-- Header -->
    <div class="space-y-4">
      <h1 class="space-y-4">
        UI Components Demo
      </h1>
      <p class="space-y-4">
        Explore our comprehensive UI component library built with Svelte stores,
        UnoCSS, PicoCSS, Melt UI, and Bits UI. All components feature
        interactive updates, modern styling, and accessibility support.
      </p>
    </div>

    <div class="space-y-4">
      <!-- Buttons Demo -->
      <Card variant="elevated" padding="lg">
        <div slot="header">
          <h2 class="space-y-4">
            Button Variants
          </h2>
          <p class="space-y-4">
            Different button styles and states with loading support
          </p>
        </div>

        <div class="space-y-4">
          <div class="space-y-4">
            <Button variant="primary">Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="secondary">Success</Button>
            <Button variant="danger">Danger</Button>
            <Button variant="danger">Warning</Button>
            <Button variant="secondary">Info</Button>
          </div>

          <div class="space-y-4">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost">Ghost</Button>
              </TooltipTrigger>
              <TooltipContent>
                <span>Ghost button (minimal style)</span>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline">Outline</Button>
              </TooltipTrigger>
              <TooltipContent>
                <span>Outline button (border only)</span>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button disabled={true}>Disabled</Button>
              </TooltipTrigger>
              <TooltipContent>
                <span>Disabled button</span>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button loadingKey="demo-loading">Loading</Button>
              </TooltipTrigger>
              <TooltipContent>
                <span>Loading button</span>
              </TooltipContent>
            </Tooltip>
          </div>

          <div class="space-y-4">
            <Button size="sm">Extra Small</Button>
            <Button size="sm">Small</Button>
            <Button size="md">Medium</Button>
            <Button size="lg">Large</Button>
            <!-- Removed unsupported xl size -->
          </div>

          <div class="space-y-4">
            <Button class="space-y-4">
              <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
              Full Width with Icon
            </Button>
            <Button
              variant="secondary"
              class="space-y-4"
            >
              <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
              Download Report
            </Button>
          </div>
        </div>
      </Card>

      <!-- Notifications Demo -->
      <Card variant="elevated" padding="lg">
        <div slot="header">
          <h2 class="space-y-4">
            Notification System
          </h2>
          <p class="space-y-4">
            Toast notifications with actions and auto-dismiss
          </p>
        </div>

        <div class="space-y-4">
          <Button
            variant="secondary"
            class="space-y-4"
            onclick={() => showSuccessNotification()}
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Show Success
          </Button>
          <Button
            variant="danger"
            class="space-y-4"
            onclick={() => showErrorNotification()}
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Show Error
          </Button>
          <Button
            variant="danger"
            class="space-y-4"
            onclick={() => showWarningNotification()}
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Show Warning
          </Button>
          <Button
            variant="secondary"
            class="space-y-4"
            onclick={() => showInfoNotification()}
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Show Info
          </Button>
        </div>
      </Card>

      <!-- Modals Demo -->
      <Card variant="elevated" padding="lg">
        <div slot="header">
          <h2 class="space-y-4">
            Modal System
          </h2>
          <p class="space-y-4">
            Confirm dialogs, alerts, and prompts
          </p>
        </div>

        <div class="space-y-4">
          <button
            class="space-y-4"
            type="button"
            onclick={() => showConfirmModal()}
            aria-label="Confirm Dialog"
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Confirm Dialog
          </button>
          <button
            class="space-y-4"
            type="button"
            onclick={() => showAlertModal()}
            aria-label="Alert Dialog"
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Alert Dialog
          </button>
          <button
            class="space-y-4"
            type="button"
            onclick={() => showPromptModal()}
            aria-label="Prompt Dialog"
          >
            <iconify-icon data-icon="${1}" class="space-y-4"></iconify-icon>
            Prompt Dialog
          </button>
        </div>
      </Card>

      <!-- Interactive Cards Demo -->
      <Card variant="elevated" padding="lg">
        <div slot="header">
          <h2 class="space-y-4">
            Interactive Cards
          </h2>
          <p class="space-y-4">
            Cards with hover effects and selection states
          </p>
        </div>

        <div class="space-y-4">
          <Card
            variant="outlined"
            interactive
            selected={interactiveCardSelected}
            onclick={() =>
              (interactiveCardSelected = !interactiveCardSelected)}
            padding="md"
          >
            <div class="space-y-4">
              <div>
                <h3 class="space-y-4">
                  Clickable Card
                </h3>
                <p class="space-y-4">
                  Click to select/deselect
                </p>
              </div>
              <div class="space-y-4">
                {interactiveCardSelected ? "✅" : "⭕"}
              </div>
            </div>
          </Card>

          <Card variant="filled" padding="md">
            <div slot="header">
              <h3 class="space-y-4">
                Card with Slots
              </h3>
            </div>

            <p class="space-y-4">
              This card demonstrates the header and footer slots.
            </p>

            <div slot="footer">
              <div
                class="space-y-4"
              >
                <span>Footer content</span>
                <span>Created today</span>
              </div>
            </div>
          </Card>

          <Card variant="elevated" loading padding="md">
            <p class="space-y-4">
              This card is in a loading state...
            </p>
          </Card>
        </div>
      </Card>

      <!-- Form Demo -->
      <div class="space-y-4">
        <Card variant="elevated" padding="lg">
          <div slot="header">
            <h2 class="space-y-4">
              Advanced Form System
            </h2>
            <p class="space-y-4">
              Forms with validation, error handling, and real-time feedback
            </p>
          </div>

          <Form
            options={demoFormOptions}
            onsubmit={handleDemoFormSubmit}
            submitText="Create Account"
            showResetButton={true}
            resetText="Clear Form"
            class="space-y-4"
          >
            <div
              slot="default"
              let:form
              let:formApi
              let:values
              let:errors
              let:isValid
              let:isDirty
            >
              <div class="space-y-4">
                <h3
                  class="space-y-4"
                >
                  Account Information
                </h3>
              </div>

              <Input
                label="Full Name"
                placeholder="Enter your full name"
                required
                data-icon="${1}"
                value={values.name || ""}
                error={errors.name}
                oninput={(e) =>
                  formApi.setField(
                    "name",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi.touchField("name")}
              />

              <Input
                label="Email Address"
                type="email"
                placeholder="Enter your email"
                required
                data-icon="${1}"
                value={values.email || ""}
                error={errors.email}
                success={values.email && !errors.email}
                oninput={(e) =>
                  formApi.setField(
                    "email",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi.touchField("email")}
              />

              <Input
                label="Password"
                type="password"
                placeholder="Create a password"
                required
                data-icon="${1}"
                value={values.password || ""}
                error={errors.password}
                hint="Must be at least 8 characters long"
                oninput={(e) =>
                  formApi.setField(
                    "password",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi.touchField("password")}
              />

              <Input
                label="Confirm Password"
                type="password"
                placeholder="Confirm your password"
                required
                data-icon="${1}"
                value={values.confirmPassword || ""}
                error={errors.confirmPassword}
                oninput={(e) =>
                  formApi.setField(
                    "confirmPassword",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi.touchField("confirmPassword")}
              />

              <div class="space-y-4">
                <label
                  class="space-y-4"
                >
                  <input
                    type="checkbox"
                    class="space-y-4"
                    checked={values.terms || false}
                    onchange={(e) =>
                      formApi.setField(
                        "terms",
                        (e.target as HTMLInputElement)?.checked
                      )}
                    onblur={() => formApi.touchField("terms")}
                  />
                  I agree to the terms and conditions *
                </label>
                {#if errors.terms}
                  <p class="space-y-4">
                    {errors.terms}
                  </p>
                {/if}
              </div>
            </div>
          </Form>
        </Card>
      </div>
    </div>

    <!-- Component Features -->
    <Card variant="outlined" padding="lg" class="space-y-4">
      <div slot="header">
        <h2 class="space-y-4">
          🚀 Component Features
        </h2>
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <h3
            class="space-y-4"
          >
            <iconify-icon data-icon="${1}" class="space-y-4"
            ></iconify-icon>
            Modern Styling
          </h3>
          <ul class="space-y-4">
            <li>• UnoCSS utility classes</li>
            <li>• PicoCSS base styles</li>
            <li>• Dark mode support</li>
            <li>• Responsive design</li>
          </ul>
        </div>

        <div class="space-y-4">
          <h3
            class="space-y-4"
          >
            <iconify-icon data-icon="${1}" class="space-y-4"
            ></iconify-icon>
            Interactive Updates
          </h3>
          <ul class="space-y-4">
            <li>• Svelte stores integration</li>
            <li>• Real-time state management</li>
            <li>• Optimistic updates</li>
            <li>• Smooth animations</li>
          </ul>
        </div>

        <div class="space-y-4">
          <h3
            class="space-y-4"
          >
            <iconify-icon data-icon="${1}" class="space-y-4"
            ></iconify-icon>
            Accessibility
          </h3>
          <ul class="space-y-4">
            <li>• ARIA labels and roles</li>
            <li>• Keyboard navigation</li>
            <li>• Screen reader support</li>
            <li>• Focus management</li>
          </ul>
        </div>
      </div>
    </Card>
  </div>
</div>
