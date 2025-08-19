<script lang="ts">
  interface Props {
    form: ActionData;
  }
  let {
    form
  }: Props = $props();



  import { goto } from "$app/navigation";
  import EnhancedCaseForm from "$lib/components/forms/EnhancedCaseForm.svelte";
  import { notifications } from "$lib/stores/notification";
  import TauriAPI from "$lib/tauri";
  import type { ActionData } from "./$types";

  
  let isSubmitting = false;
  let caseData = {};

  async function handleSubmit(event: CustomEvent) {
    const { data } = event.detail;
    isSubmitting = true;

    try {
      // Call Tauri API to create case
      const newCase = await TauriAPI.createCase(data);

      notifications.add({
        type: "success",
        title: "Case Created",
        message: `Case "${data.title}" has been created successfully.`,
      });

      // Redirect to the new case
      await goto(`/cases/${newCase.id}`);
    } catch (error) {
      console.error("Failed to create case:", error);
      notifications.add({
        type: "error",
        title: "Failed to Create Case",
        message: "There was an error creating the case. Please try again.",
      });
    } finally {
      isSubmitting = false;
}}
  function handleCancel() {
    goto("/cases");
}
  // Show server-side form errors as notifications
  $effect(() => {
    if (form?.error) {
      notifications.error("Form Error", form.error);
    }
  });
</script>

<svelte:head>
  <title>Create New Case - WardenNet Detective Mode</title>
  <meta
    name="description"
    content="Create a new investigation case with comprehensive validation and security features"
  />
</svelte:head>

<div class="max-w-4xl mx-auto p-6 bg-gray-50 min-h-screen">
  <!-- Header -->
  <div class="bg-white rounded-lg shadow-sm p-6 mb-6">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">Create New Case</h1>
    <p class="text-gray-600">
      Build a comprehensive case file with evidence and documentation
    </p>
  </div>

  <!-- Enhanced Case Form -->
  <EnhancedCaseForm
    case_={caseData}
    user={{
      id: "1",
      name: "Current User",
      email: "user@example.com",
      role: "admin",
      isActive: true,
      createdAt: new Date(),
      updatedAt: new Date(),
    }}
    on:submit={handleSubmit}
    oncancel={handleCancel}
  />
</div>
