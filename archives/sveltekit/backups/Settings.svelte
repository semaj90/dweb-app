<script lang="ts">
  export let isOpen: boolean;
  export let settings: {
    theme: string;
    language: string;
    ttsEngine: string;
    voiceLanguage: string;
    enableSuggestions: boolean;
    enableMasking: boolean;
    enableAutoSave: boolean;
    maxHistoryItems: number;
    enableNotifications: boolean;
    fontFamily: string;
    fontSize: string;
  };
</script>

{#if isOpen}
  <div className="${1}">
    <div className="${1}">
      <h2>Settings</h2>
      <form>
        <label>
          Theme:
          <select bind:value={settings.theme}>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>

        <label>
          Language:
          <select bind:value={settings.language}>
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <!-- Add more languages as needed -->
          </select>
        </label>

        <label>
          TTS Engine:
          <select bind:value={settings.ttsEngine}>
            <option value="default">Default</option>
            <option value="engine1">Engine 1</option>
            <!-- Add more TTS engines as needed -->
          </select>
        </label>

        <label>
          Voice Language:
          <select bind:value={settings.voiceLanguage}>
            <option value="en-US">English (US)</option>
            <option value="en-GB">English (UK)</option>
            <option value="es-ES">Spanish (Spain)</option>
            <!-- Add more voice languages as needed -->
          </select>
        </label>

        <label>
          Enable Suggestions:
          <input type="checkbox" bind:checked={settings.enableSuggestions} />
        </label>

        <label>
          Enable Masking:
          <input type="checkbox" bind:checked={settings.enableMasking} />
        </label>

        <label>
          Enable Auto-Save:
          <input type="checkbox" bind:checked={settings.enableAutoSave} />
        </label>

        <label>
          Max History Items:
          <input type="number" bind:value={settings.maxHistoryItems} />
        </label>

        <label>
          Enable Notifications:
          <input type="checkbox" bind:checked={settings.enableNotifications} />
        </label>

        <label>
          Font Family:
          <select bind:value={settings.fontFamily}>
            <option value="Arial">Arial</option>
            <option value="Courier New">Courier New</option>
            <option value="Georgia">Georgia</option>
            <option value="Times New Roman">Times New Roman</option>
            <!-- Add more font families as needed -->
          </select>
        </label>

        <label>
          Font Size:
          <input type="text" bind:value={settings.fontSize} placeholder="e.g., 16px" />
        </label>

        <button type="submit">Save Settings</button>
      </form>
    </div>
  </div>
{/if}