<script lang="ts">
    import { createEventDispatcher } from "svelte";

    const dispatch = createEventDispatcher();

    export interface CaseItem {
        id: string
        title: string
        description?: string;
        status?: "open" | "pending" | "closed";
        createdAt?: string; // ISO date or human string
        tags?: string[];
    }

    export let caseItem: CaseItem = { id: "", title: "" };
    export let className = "";

    function handleEdit() {
        dispatch("edit", caseItem);
    }

    function handleDelete() {
        dispatch("delete", caseItem);
    }
</script>

<article class={"bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md p-4 shadow-sm " + className} aria-labelledby={"case-" + caseItem.id}>
    <header class="flex items-start justify-between space-x-4">
        <div>
            <h3 id={"case-" + caseItem.id} class="text-sm font-medium text-gray-900 dark:text-gray-100">
                {caseItem.title}
            </h3>
            {#if caseItem.createdAt}
                <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">{new Date(caseItem.createdAt).toLocaleString()}</p>
            {/if}
        </div>

        <div class="flex items-center space-x-2">
            {#if caseItem.status}
                <span
                    class="text-xs px-2 py-0.5 rounded-full font-medium"
                    class:selected-open={caseItem.status === "open"}
                    class:selected-pending={caseItem.status === "pending"}
                    class:selected-closed={caseItem.status === "closed"}
                >
                    {caseItem.status}
                </span>
            {/if}

            <div class="flex space-x-1">
                <button
                    type="button"
                    class="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                    on:click={handleEdit}
                >
                    Edit
                </button>
                <button
                    type="button"
                    class="text-xs px-2 py-1 bg-red-600 text-white rounded hover:bg-red-700"
                    on:click={handleDelete}
                >
                    Delete
                </button>
            </div>
        </div>
    </header>

    {#if caseItem.description}
        <p class="mt-3 text-sm text-gray-700 dark:text-gray-300">{caseItem.description}</p>
    {/if}

    {#if caseItem.tags && caseItem.tags.length}
        <footer class="mt-3 flex flex-wrap gap-2">
            {#each caseItem.tags as t}
                <span class="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded">
                    {t}
                </span>
            {/each}
        </footer>
    {/if}
</article>

<style>
    /* small status color helpers */
    .selected-open { background: #ecfdf5; color: #065f46; } /* green-50 / green-700 */
    .selected-pending { background: #fff7ed; color: #92400e; } /* orange-50 / orange-700 */
    .selected-closed { background: #f8fafc; color: #475569; } /* slate-50 / slate-600 */
</style>