<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  let notesContent: string = '';
  let selectedCaseForNotes: string = ''; // Assuming notes can be linked to a case
  let selectedPoiForNotes: string = ''; // Assuming notes can be linked to a POI

  // Dummy data for dropdowns - replace with actual data fetched from API
  const caseOptions = [
    { value: 'case1', label: 'Case 2023-001' },
    { value: 'case2', label: 'Case 2023-002' },
    { value: 'case3', label: 'Case 2023-003' },
  ];

  const poiOptions = [
    { value: 'poi1', label: 'John Doe' },
    { value: 'poi2', label: 'Jane Smith' },
    { value: 'poi3', label: 'Criminal X' },
  ];

  const handleSubmit = async () => {
    // In a real application, you would send this data to your backend API
    // For now, we'll just log it and dispatch an event.
    console.log('Add Notes Data:', {
      notesContent,
      selectedCaseForNotes,
      selectedPoiForNotes,
    });

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    alert('Notes saved successfully!');
    dispatch('notesSaved');

    // Reset form
    notesContent = '';
    selectedCaseForNotes = '';
    selectedPoiForNotes = '';
  };
</script>

<div class="card">
  <div class="card-header">
    <h3>Add Notes</h3>
  </div>
  <div class="card-body">
    <div class="mb-3">
      <label for="notesContent" class="form-label">Notes:</label>
      <textarea id="notesContent" class="form-control" bind:value={notesContent} rows="5"></textarea>
    </div>
    <div class="mb-3">
      <label for="caseSelectNotes" class="form-label">Link to Case (Optional):</label>
      <select id="caseSelectNotes" class="form-control" bind:value={selectedCaseForNotes}>
        <option value="">Select a case</option>
        {#each caseOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>
    <div class="mb-3">
      <label for="poiSelectNotes" class="form-label">Link to POI (Optional):</label>
      <select id="poiSelectNotes" class="form-control" bind:value={selectedPoiForNotes}>
        <option value="">Select a POI</option>
        {#each poiOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>
    <button class="btn btn-primary" on:click={handleSubmit}>Save Notes</button>
  </div>
</div>

<style>
  .card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
  }

  .card-header {
    border-bottom: 1px solid #eee;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
  }

  .card-header h3 {
    margin: 0;
    font-size: 1.25rem;
    color: #333;
  }

  .form-label {
    font-weight: bold;
    margin-bottom: 0.5rem;
    display: block;
  }

  .form-control {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
  }

  textarea.form-control {
    resize: vertical;
  }

  .btn-primary {
    background-color: #007bff;
    color: #fff;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
  }

  .btn-primary:hover {
    background-color: #0056b3;
  }
</style>