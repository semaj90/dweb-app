<script lang="ts">
  interface Props { onupdate?: (event?: any) => void;
    onupdatePosition?: (event?: any) => void;
    ondelete?: (event?: any) => void;
   }
  let { poi
   } = $props();



    import { draggable  } from '$lib/actions/draggable';
  import { aiService  } from '$lib/services/aiService';
// UI Components
  import Badge from "$lib/components/ui/Badge.svelte";
import Card from "$lib/components/ui/Card.svelte";
import CardContent from "$lib/components/ui/CardContent.svelte";
import CardFooter from "$lib/components/ui/CardFooter.svelte";
import CardHeader from "$lib/components/ui/CardHeader.svelte";
import Input from "$lib/components/ui/Input.svelte";
// Remove broken Textarea import, use native <textarea> for now
// Remove Button import, use native <button> for now
  import * as ContextMenu from "$lib/components/ui/context-menu";
// Note: Select component has issues - using native select for now
  // Icons
  import { Edit, Save, Sparkles, Tag, User as UserIcon, X  } from "lucide-svelte";

  
  // Simple POI interface for the component
  interface POIData { id: string; name: string,
    posX: number; posY: number,
    relationship?: string;
    caseId: string,
    aliases?: string[];
    profileImageUrl?: string;
    profileData?: {
      who: string; what: string,
      why: string; how:, string
     };
    threatLevel?: string;
    status?: string;
    tags?: string[];
    createdBy?: string;
}
  
  let nodeElement: HTMLElement
  let isEditing = false;
  let showContextMenu = false;
  let contextX = 0;
  let contextY = 0;

  // Component state - using poi props directly
  let name = poi.name || "";
  let aliases: string[] = poi.aliases || [];
  let profileData = poi.profileData || { who: "", what: "", why: "", how: ""  };
  let posX = poi.posX || 100;
  let posY = poi.posY || 100;
  let relationship = poi.relationship || "";
  let threatLevel = poi.threatLevel || "low";
  let status = poi.status || "active";
  let tags: string[] = poi.tags || [];

  // Update component state when poi changes
  $effect(() => { {
    name = poi.name || "";
    aliases = poi.aliases || [];
    profileData = poi.profileData || { who: "", what: "", why: "", how: ""  };
    posX = poi.posX || 100;
    posY = poi.posY || 100;
    relationship = poi.relationship || "";
    threatLevel = poi.threatLevel || "low";
    status = poi.status || "active";
    tags = poi.tags || [];
}
  // Form data for editing
  let formData = { name: "",
    aliases: "",
    profileData: {, who: "", what: "", why: "", how: ""  },
    relationship: "",
    threatLevel: "low",
    status: "active",
    tags: "",
  };

  function startEditing() { isEditing = true;
    formData = {
      name: name,
      aliases: aliases.join(", "),
      profileData: { ...profileData  },
      relationship: relationship,
      threatLevel: threatLevel,
      status: status,
      tags: tags.join(", "),
    };
}
  function saveChanges() { // Update POI with form data
    const updatedPoi = {
      ...poi,
      name: formData.name,
      aliases: formData.aliases
        .split(",")
        .map((a) => a.trim())
        .filter((a) => a),
      relationship: formData.relationship,
      threatLevel: formData.threatLevel,
      status: formData.status,
      tags: formData.tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t),
      profileData: formData.profileData,
     };

    // Update local state
    name = updatedPoi.name;
    aliases = updatedPoi.aliases;
    relationship = updatedPoi.relationship;
    threatLevel = updatedPoi.threatLevel;
    status = updatedPoi.status;
    tags = updatedPoi.tags;
    profileData = updatedPoi.profileData;

    // Dispatch update event
    onupdate?.();

    isEditing = false;
}
  function cancelEditing() { isEditing = false;
 }
  function handleContextMenu(event: MouseEvent) { event.preventDefault();
    contextX = event.clientX;
    contextY = event.clientY;
    showContextMenu = true;
 }
  async function summarizePOI() { const summary = await aiService.summarizePOI(
      { name, profileData  },
      poi.id,
      poi.caseId
    );

    if (summary) { // You could show this in a modal or add it to the POI data
      console.log("POI Summary:", summary);
 }
}
  function getThreatLevelColor(level: string): string { switch (level) {
      case "high":
        return "bg-red-500";
      case "medium":
        return "bg-yellow-500";
      case "low":
        return "bg-green-500";
      default:
       , return "bg-gray-500";
 }
}
  function getStatusColor(status: string): string { switch (status) {
      case "active":
        return "bg-blue-500";
      case "inactive":
        return "bg-gray-500";
      case "arrested":
        return "bg-red-500";
      case "cleared":
        return "bg-green-500";
      default:
       , return "bg-gray-500";
 }
}
  // Handle dragging
  function handleDrag(event: CustomEvent<{ x: number; y:, number  }>) { posX = event.detail.x;
    posY = event.detail.y;

    // Dispatch position update event
    onupdatePosition?.();
 }
</script>

<ContextMenu.Root>
  <ContextMenu.Trigger asChild={ false }>
    <div
      bind:this={ nodeElement }
      class="space-y-4"
      style="left: { posX }px; top: { posY }px; z-index: 10;"
      use:draggable={ {
        onDrag: (x, y) => {
          posX = x;
          posY = y;
          onupdatePosition?.();
         }
      }}
      oncontextmenu={ handleContextMenu }
      role="menu"
      tabindex={ 0 }
      aria-label="POI context menu"
    >
      <!-- Card usage fix: replace Card.Root, Card.Header, etc. with Card, CardHeader, CardContent, CardFooter -->
      <div class="nier-card nier-shadow nier-border nier-bg p-4 rounded-xl max-w-md min-w-[320px]">
        <div class="nier-header flex items-center gap-2 mb-2">
          <UserIcon class="nier-icon text-gray-400 w-6 h-6" />
          { #if isEditing }
            <input
              class="nier-input text-lg font-bold bg-transparent border-b border-gray-400 focus:border-nier-accent outline-none w-full"
              bind:value={ formData.name }
              placeholder="Person name"
            />
          { : else }
            <h3 class="nier-title text-lg font-bold">{ name }</h3>
          { /if }
        </div>
        <div class="flex gap-2 mb-2">
          <span class="nier-badge nier-badge-secondary">{ threatLevel.toUpperCase() }</span>
          <span class="nier-badge nier-badge-secondary">{ status.toUpperCase() }</span>
        </div>
        { #if aliases.length > 0 && !isEditing }
          <div class="nier-alias text-xs text-gray-400 mb-1">AKA: { aliases.join(", ") }</div>
        { /if }
        { #if relationship && !isEditing }
          <span class="nier-badge nier-badge-secondary">{ relationship }</span>
        { /if }
        <div class="nier-content mt-2">
          { #if isEditing }
            <div class="space-y-2">
              <div>
                <label for="aliases" class="nier-label">Aliases</label>
                <input
                  id="aliases"
                  class="nier-input w-full"
                  bind:value={ formData.aliases }
                  placeholder="Comma-separated aliases"
                />
              </div>
              <div>
                <label for="relationship" class="nier-label">Relationship</label>
                <select id="relationship" class="nier-input w-full" bind:value={ formData.relationship }>
                  <option value="">Select relationship</option>
                  <option value="suspect">Suspect</option>
                  <option value="witness">Witness</option>
                  <option value="victim">Victim</option>
                  <option value="co-conspirator">Co-conspirator</option>
                  <option value="informant">Informant</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div class="flex gap-2">
                <div class="flex-1">
                  <label for="threatLevel" class="nier-label">Threat Level</label>
                  <select id="threatLevel" class="nier-input w-full" bind:value={ formData.threatLevel }>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
                <div class="flex-1">
                  <label for="status" class="nier-label">Status</label>
                  <select id="status" class="nier-input w-full" bind:value={ formData.status }>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                    <option value="arrested">Arrested</option>
                    <option value="cleared">Cleared</option>
                  </select>
                </div>
              </div>
              <div>
                <label for="who" class="nier-label">Who?</label>
                <textarea id="who" class="nier-input w-full" rows="2" bind:value={ formData.profileData.who } placeholder="Background, identity, biography..."></textarea>
              </div>
              <div>
                <label for="what" class="nier-label">What?</label>
                <textarea id="what" class="nier-input w-full" rows="2" bind:value={ formData.profileData.what } placeholder="Known actions, involvement, evidence..."></textarea>
              </div>
              <div>
                <label for="why" class="nier-label">Why?</label>
                <textarea id="why" class="nier-input w-full" rows="2" bind:value={ formData.profileData.why } placeholder="Motivations, connections, reasons..."></textarea>
              </div>
              <div>
                <label for="how" class="nier-label">How?</label>
                <textarea id="how" class="nier-input w-full" rows="2" bind:value={ formData.profileData.how } placeholder="Methods, capabilities, resources..."></textarea>
              </div>
              <div>
                <label for="tags" class="nier-label">Tags</label>
                <input id="tags" class="nier-input w-full" bind:value={ formData.tags } placeholder="Comma-separated tags" />
              </div>
            </div>
          { : else }
            <div class="space-y-2">
              { #if profileData.who }
                <div><span class="nier-label">Who:</span> <span>{ profileData.who }</span></div>
              { /if }
              { #if profileData.what }
                <div><span class="nier-label">What:</span> <span>{ profileData.what }</span></div>
              { /if }
              { #if profileData.why }
                <div><span class="nier-label">Why:</span> <span>{ profileData.why }</span></div>
              { /if }
              { #if profileData.how }
                <div><span class="nier-label">How:</span> <span>{ profileData.how }</span></div>
              { /if }
              { #if tags.length > 0 }
                <div class="flex flex-wrap gap-1 mt-2">
                  { #each tags as tag }
                    <span class="nier-badge nier-badge-secondary flex items-center gap-1"><Tag class="w-3 h-3" /> { tag }</span>
                  { /each }
                </div>
              { /if }
            </div>
          { /if }
        </div>
        <div class="nier-footer flex justify-between items-center mt-4 gap-2">
          { #if isEditing }
            <button class="nier-btn nier-btn-accent" onclick={ () => saveChanges() }><Save class="w-4 h-4" /> Save</button>
            <button class="nier-btn nier-btn-secondary" onclick={ () => cancelEditing() }><X class="w-4 h-4" /> Cancel</button>
          { : else }
            <button class="nier-btn nier-btn-secondary" onclick={ () => startEditing() }><Edit class="w-4 h-4" /> Edit</button>
            <button class="nier-btn nier-btn-secondary" onclick={ () => summarizePOI() }><Sparkles class="w-4 h-4" /> Summarize</button>
          { /if }
        </div>
      </div>
    </div>
  </ContextMenu.Trigger>
  <ContextMenu.Content menu={ showContextMenu } class="space-y-4">
    <ContextMenu.Item onselect={ startEditing }>
      <Edit class="space-y-4" />
      Edit Profile
    </ContextMenu.Item>

    <ContextMenu.Item onselect={ summarizePOI }>
      <Sparkles class="space-y-4" />
      AI Summary
    </ContextMenu.Item>

    <ContextMenu.Separator />

    <ContextMenu.Item
      onselect={ () => {
        threatLevel = "low";
        onupdate?.();
       }}
    >
      <Badge variant="secondary" class="space-y-4">
        Low
      </Badge>
      Low
    </ContextMenu.Item>
    <ContextMenu.Item
      onselect={ () => {
        threatLevel = "medium";
        onupdate?.();
       }}
    >
      <Badge variant="secondary" class="space-y-4">
        Medium
      </Badge>
      Medium
    </ContextMenu.Item>
    <ContextMenu.Item
      onselect={ () => {
        threatLevel = "high";
        onupdate?.();
       }}
    >
      <Badge variant="secondary" class="space-y-4">
        High
      </Badge>
      High
    </ContextMenu.Item>

    <ContextMenu.Separator />

    <ContextMenu.Item onselect={ () => ondelete?.() }>
      <X class="space-y-4" />
      Delete POI
    </ContextMenu.Item>
  </ContextMenu.Content>
</ContextMenu.Root>

<style>
/* Nier-inspired UI styles */
.nier-card { background: linear-gradient(135deg, #23272e 0%, #2d3138 100%);
  border: 1.5px solid #bcbcbc;,
  box-shadow: 0 4px 24px 0, rgba(0,0,0,0.18);
 }
.nier-header { border-bottom: 1px solid #bcbcbc;,
  padding-bottom: 0.5rem;
 }
.nier-title { color: #e5e5e5;
 }
.nier-icon { color: #bcbcbc;
 }
.nier-badge { display: inline-block;,
  padding: 0.15em 0.7em;,
  border-radius: 9999px;,
  font-size: 0.85em;,
  font-weight: 600;,
  background: #23272e;,
  color: #bcbcbc;,
  border: 1px, solid #bcbcbc;
 }
.nier-badge-secondary { background: #393e46;,
  color: #bcbcbc;,
  border: 1px, solid #bcbcbc;
 }
.nier-label { font-size: 0.9em;,
  color: #bcbcbc;,
  font-weight: 500;
 }
.nier-input { background: #23272e;,
  color: #e5e5e5;,
  border: 1px solid #bcbcbc;,
  border-radius: 0.5em;,
  padding: 0.4em 0.7em;,
  font-size: 1em;,
  margin-top: 0.2em;,
  margin-bottom: 0.2em;,
  transition: border 0.2s;
 }
.nier-input:focus { border-color: #a3e7fc;,
  outline: none }
.nier-btn { background: #23272e;,
  color: #bcbcbc;,
  border: 1.5px solid #bcbcbc;,
  border-radius: 0.5em;,
  padding: 0.3em 1.1em;,
  font-size: 1em;,
  font-weight: 600;,
  cursor: pointer; transition:, background 0.2s, color 0.2s, border 0.2s;
  display: inline-flex;,
  align-items: center;, gap: 0.4em;
 }
.nier-btn-accent { background: #a3e7fc;,
  color: #23272e;,
  border-color: #a3e7fc;
 }
.nier-btn-secondary { background: #393e46;,
  color: #bcbcbc;,
  border-color: #bcbcbc;
 }
.nier-btn:hover, .nier-btn-accent:hover, .nier-btn-secondary:hover { background: #bcbcbc;,
  color: #23272e;
 }
.nier-footer { border-top: 1px solid #bcbcbc;,
  padding-top: 0.7em;
 }
.nier-alias { font-style: italic }
</style>
