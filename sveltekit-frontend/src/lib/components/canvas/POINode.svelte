<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { draggable } from '$lib/actions/draggable';
  import { aiService } from '$lib/services/aiService';
// UI Components
  import Badge from "$lib/components/ui/Badge.svelte";
import Card from "$lib/components/ui/Card.svelte";
import CardContent from "$lib/components/ui/CardContent.svelte";
import CardFooter from "$lib/components/ui/CardFooter.svelte";
import CardHeader from "$lib/components/ui/CardHeader.svelte";
import Input from "$lib/components/ui/Input.svelte";
import Textarea from "$lib/components/ui/Textarea.svelte";
  import Button from "$lib/components/ui/button";
  import * as ContextMenu from "$lib/components/ui/context-menu";
// Note: Select component has issues - using native select for now
  // Icons
  import { Edit, Save, Sparkles, Tag, User as UserIcon, X } from "lucide-svelte";

  const dispatch = createEventDispatcher();

  // Simple POI interface for the component
  interface POIData {
    id: string;
    name: string;
    posX: number;
    posY: number;
    relationship?: string;
    caseId: string;
    aliases?: string[];
    profileImageUrl?: string;
    profileData?: {
      who: string;
      what: string;
      why: string;
      how: string;
    };
    threatLevel?: string;
    status?: string;
    tags?: string[];
    createdBy?: string;
}
  export let poi: POIData;

  let nodeElement: HTMLElement;
  let isEditing = false;
  let showContextMenu = false;
  let contextX = 0;
  let contextY = 0;

  // Component state - using poi props directly
  let name = poi.name || "";
  let aliases: string[] = poi.aliases || [];
  let profileData = poi.profileData || { who: "", what: "", why: "", how: "" };
  let posX = poi.posX || 100;
  let posY = poi.posY || 100;
  let relationship = poi.relationship || "";
  let threatLevel = poi.threatLevel || "low";
  let status = poi.status || "active";
  let tags: string[] = poi.tags || [];

  // Update component state when poi changes
  $: {
    name = poi.name || "";
    aliases = poi.aliases || [];
    profileData = poi.profileData || { who: "", what: "", why: "", how: "" };
    posX = poi.posX || 100;
    posY = poi.posY || 100;
    relationship = poi.relationship || "";
    threatLevel = poi.threatLevel || "low";
    status = poi.status || "active";
    tags = poi.tags || [];
}
  // Form data for editing
  let formData = {
    name: "",
    aliases: "",
    profileData: { who: "", what: "", why: "", how: "" },
    relationship: "",
    threatLevel: "low",
    status: "active",
    tags: "",
  };

  function startEditing() {
    isEditing = true;
    formData = {
      name: name,
      aliases: aliases.join(", "),
      profileData: { ...profileData },
      relationship: relationship,
      threatLevel: threatLevel,
      status: status,
      tags: tags.join(", "),
    };
}
  function saveChanges() {
    // Update POI with form data
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
    dispatch("update", updatedPoi);

    isEditing = false;
}
  function cancelEditing() {
    isEditing = false;
}
  function handleContextMenu(event: MouseEvent) {
    event.preventDefault();
    contextX = event.clientX;
    contextY = event.clientY;
    showContextMenu = true;
}
  async function summarizePOI() {
    const summary = await aiService.summarizePOI(
      { name, profileData },
      poi.id,
      poi.caseId
    );

    if (summary) {
      // You could show this in a modal or add it to the POI data
      console.log("POI Summary:", summary);
}
}
  function getThreatLevelColor(level: string): string {
    switch (level) {
      case "high":
        return "bg-red-500";
      case "medium":
        return "bg-yellow-500";
      case "low":
        return "bg-green-500";
      default:
        return "bg-gray-500";
}
}
  function getStatusColor(status: string): string {
    switch (status) {
      case "active":
        return "bg-blue-500";
      case "inactive":
        return "bg-gray-500";
      case "arrested":
        return "bg-red-500";
      case "cleared":
        return "bg-green-500";
      default:
        return "bg-gray-500";
}
}
  // Handle dragging
  function handleDrag(event: CustomEvent<{ x: number; y: number }>) {
    posX = event.detail.x;
    posY = event.detail.y;

    // Dispatch position update event
    dispatch("updatePosition", { id: poi.id, x: posX, y: posY });
}
</script>

<ContextMenu.Root>
  <ContextMenu.Trigger asChild={false}>
    <div
      bind:this={nodeElement}
      class="container mx-auto px-4"
      style="left: {posX}px; top: {posY}px; z-index: 10;"
      use:draggable={{
        onDrag: (x, y) => {
          posX = x;
          posY = y;
          dispatch("updatePosition", { id: poi.id, x: posX, y: posY });
        },
      "
      on:contextmenu={handleContextMenu}
      role="menu"
      tabindex={0}
      aria-label="POI context menu"
    >
      <!-- Card usage fix: replace Card.Root, Card.Header, etc. with Card, CardHeader, CardContent, CardFooter -->
      <Card class="container mx-auto px-4">
        <CardHeader class="pb-2">
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <UserIcon class="container mx-auto px-4" />
              {#if isEditing}
                <Input
                  bind:value={formData.name}
                  class="container mx-auto px-4"
                  placeholder="Person name"
                />
              {:else}
                <h3 class="container mx-auto px-4">{name}</h3>
              {/if}
            </div>
            <div class="container mx-auto px-4">
              <Badge
                variant="secondary"
                class="container mx-auto px-4"
              >
                {threatLevel.toUpperCase()}
              </Badge>
              <Badge
                variant="secondary"
                class="container mx-auto px-4"
              >
                {status.toUpperCase()}
              </Badge>
            </div>
          </div>
          {#if aliases.length > 0 && !isEditing}
            <div class="container mx-auto px-4">
              AKA: {aliases.join(", ")}
            </div>
          {/if}
          {#if relationship && !isEditing}
            <Badge variant="secondary" class="container mx-auto px-4">
              {relationship}
            </Badge>
          {/if}
        </CardHeader>
        <CardContent class="space-y-3">
          {#if isEditing}
            <!-- Edit Form -->
            <div class="container mx-auto px-4">
              <div>
                <label for="aliases" class="container mx-auto px-4"
                  >Aliases</label
                >
                <Input
                  id="aliases"
                  bind:value={formData.aliases}
                  placeholder="Comma-separated aliases"
                  class="container mx-auto px-4"
                />
              </div>
              <div>
                <label
                  for="relationship"
                  class="container mx-auto px-4">Relationship</label
                >
                <select
                  id="relationship"
                  bind:value={formData.relationship}
                  class="container mx-auto px-4"
                >
                  <option value="">Select relationship</option>
                  <option value="suspect">Suspect</option>
                  <option value="witness">Witness</option>
                  <option value="victim">Victim</option>
                  <option value="co-conspirator">Co-conspirator</option>
                  <option value="informant">Informant</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div class="container mx-auto px-4">
                <div>
                  <label
                    for="threatLevel"
                    class="container mx-auto px-4"
                    >Threat Level</label
                  >
                  <select
                    id="threatLevel"
                    bind:value={formData.threatLevel}
                    class="container mx-auto px-4"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
                <div>
                  <label for="status" class="container mx-auto px-4"
                    >Status</label
                  >
                  <select
                    id="status"
                    bind:value={formData.status}
                    class="container mx-auto px-4"
                  >
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                    <option value="arrested">Arrested</option>
                    <option value="cleared">Cleared</option>
                  </select>
                </div>
              </div>
              <!-- Profile Data (Who, What, Why, How) -->
              <div class="container mx-auto px-4">
                <div>
                  <label for="who" class="container mx-auto px-4"
                    >Who?</label
                  >
                  <Textarea
                    id="who"
                    bind:value={formData.profileData.who}
                    placeholder="Background, identity, biography..."
                    class="container mx-auto px-4"
                    rows="4"
                  />
                </div>
                <div>
                  <label for="what" class="container mx-auto px-4"
                    >What?</label
                  >
                  <Textarea
                    id="what"
                    bind:value={formData.profileData.what}
                    placeholder="Known actions, involvement, evidence..."
                    class="container mx-auto px-4"
                    rows="4"
                  />
                </div>
                <div>
                  <label for="why" class="container mx-auto px-4"
                    >Why?</label
                  >
                  <Textarea
                    id="why"
                    bind:value={formData.profileData.why}
                    placeholder="Motivations, connections, reasons..."
                    class="container mx-auto px-4"
                    rows="4"
                  />
                </div>
                <div>
                  <label for="how" class="container mx-auto px-4"
                    >How?</label
                  >
                  <Textarea
                    id="how"
                    bind:value={formData.profileData.how}
                    placeholder="Methods, capabilities, resources..."
                    class="container mx-auto px-4"
                    rows="4"
                  />
                </div>
              </div>
              <div>
                <label for="tags" class="container mx-auto px-4"
                  >Tags</label
                >
                <Input
                  id="tags"
                  bind:value={formData.tags}
                  placeholder="Comma-separated tags"
                  class="container mx-auto px-4"
                />
              </div>
            </div>
          {:else}
            <!-- View Mode -->
            <div class="container mx-auto px-4">
              {#if profileData.who}
                <div>
                  <div class="container mx-auto px-4">Who:</div>
                  <div class="container mx-auto px-4">{profileData.who}</div>
                </div>
              {/if}

              {#if profileData.what}
                <div>
                  <div class="container mx-auto px-4">What:</div>
                  <div class="container mx-auto px-4">{profileData.what}</div>
                </div>
              {/if}

              {#if profileData.why}
                <div>
                  <div class="container mx-auto px-4">Why:</div>
                  <div class="container mx-auto px-4">{profileData.why}</div>
                </div>
              {/if}

              {#if profileData.how}
                <div>
                  <div class="container mx-auto px-4">How:</div>
                  <div class="container mx-auto px-4">{profileData.how}</div>
                </div>
              {/if}

              {#if tags.length > 0}
                <div class="container mx-auto px-4">
                  {#each tags as tag}
                    <Badge variant="secondary" class="container mx-auto px-4">
                      <Tag class="container mx-auto px-4" />
                      {tag}
                    </Badge>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        </CardContent>
        <CardFooter class="flex justify-between items-center pt-2">
          {#if isEditing}
            <div class="container mx-auto px-4">
              <Button size="sm" on:click={() => saveChanges()}>
                <Save class="container mx-auto px-4" />
                Save
              </Button>
              <Button
                size="sm"
                variant="secondary"
                on:click={() => cancelEditing()}
              >
                <X class="container mx-auto px-4" />
                Cancel
              </Button>
            </div>
          {:else}
            <div class="container mx-auto px-4">
              <Button
                size="sm"
                variant="secondary"
                on:click={() => startEditing()}
              >
                <Edit class="container mx-auto px-4" />
                Edit
              </Button>
              <Button
                size="sm"
                variant="secondary"
                on:click={() => summarizePOI()}
              >
                <Sparkles class="container mx-auto px-4" />
                Summarize
              </Button>
            </div>
          {/if}
        </CardFooter>
      </Card>
    </div>
  </ContextMenu.Trigger>
  <ContextMenu.Content menu={showContextMenu} class="container mx-auto px-4">
    <ContextMenu.Item on:select={startEditing}>
      <Edit class="container mx-auto px-4" />
      Edit Profile
    </ContextMenu.Item>

    <ContextMenu.Item on:select={summarizePOI}>
      <Sparkles class="container mx-auto px-4" />
      AI Summary
    </ContextMenu.Item>

    <ContextMenu.Separator />

    <ContextMenu.Item
      on:select={() => {
        threatLevel = "low";
        dispatch("update", { ...poi, threatLevel: "low" });
      "
    >
      <Badge variant="secondary" class="container mx-auto px-4">
        Low
      </Badge>
      Low
    </ContextMenu.Item>
    <ContextMenu.Item
      on:select={() => {
        threatLevel = "medium";
        dispatch("update", { ...poi, threatLevel: "medium" });
      "
    >
      <Badge variant="secondary" class="container mx-auto px-4">
        Medium
      </Badge>
      Medium
    </ContextMenu.Item>
    <ContextMenu.Item
      on:select={() => {
        threatLevel = "high";
        dispatch("update", { ...poi, threatLevel: "high" });
      "
    >
      <Badge variant="secondary" class="container mx-auto px-4">
        High
      </Badge>
      High
    </ContextMenu.Item>

    <ContextMenu.Separator />

    <ContextMenu.Item on:select={() => dispatch("delete", poi.id)}>
      <X class="container mx-auto px-4" />
      Delete POI
    </ContextMenu.Item>
  </ContextMenu.Content>
</ContextMenu.Root>

<style>
  /* @unocss-include */
  .poi-node {
    transition: box-shadow 0.2s ease;
}
  .poi-node:hover {
    box-shadow: 0 4px 12px rgba(147, 51, 234, 0.15);
}
</style>
