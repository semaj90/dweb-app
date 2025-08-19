<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Dropdown from '$lib/components/+Dropdown.svelte';
  import Checkbox from '$lib/components/+Checkbox.svelte';

  const dispatch = createEventDispatcher();

  let selectedCase: string = '';
  let selectedPoi: string = '';
  let file: File | null = null;
  let summarize: boolean = false;
  let tag: boolean = false;

  const handleFileChange = (event: Event) => {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      file = input.files[0];
    } else {
      file = null;
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      alert('Please select a file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('caseId', selectedCase);
    formData.append('poiId', selectedPoi);
    formData.append('summarize', String(summarize));
    formData.append('tag', String(tag));

    try {
      const response = await fetch('/api/evidence/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        alert('File uploaded successfully!');
        dispatch('uploadSuccess');
        // Reset form
        selectedCase = '';
        selectedPoi = '';
        file = null;
        summarize = false;
        tag = false;
      } else {
        const errorData = await response.json();
        alert(`Upload failed: ${errorData.message || response.statusText}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred during file upload.');
    }
  };

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
</script>

<div class="card">
  <div class="card-header">
    <h3>Automatic File Upload</h3>
  </div>
  <div class="card-body">
    <div class="mb-3">
      <label for="caseSelect" class="form-label">Select Case:</label>
      <Dropdown id="caseSelect" bind:selected={selectedCase} options={caseOptions} />
    </div>
    <div class="mb-3">
      <label for="poiSelect" class="form-label">Select POI (Optional):</label>
      <Dropdown id="poiSelect" bind:selected={selectedPoi} options={poiOptions} />
    </div>
    <div class="mb-3">
      <label for="fileInput" class="form-label">Upload File:</label>
      <input type="file" id="fileInput" class="form-control" onchange={handleFileChange} />
    </div>
    <div class="mb-3 form-check">
      <Checkbox id="summarizeCheckbox" bind:checked={summarize} label="Summarize with AI" />
    </div>
    <div class="mb-3 form-check">
      <Checkbox id="tagCheckbox" bind:checked={tag} label="Tag with AI" />
    </div>
    <button class="btn btn-primary" onclick={handleSubmit}>Upload</button>
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