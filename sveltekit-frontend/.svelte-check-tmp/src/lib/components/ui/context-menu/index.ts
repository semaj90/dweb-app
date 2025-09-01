import Root from "./context-menu-root.svelte";
import Trigger from "./context-menu-trigger.svelte";
import Content from "./context-menu-content.svelte";
import Item from "./context-menu-item.svelte";
import Separator from "./context-menu-separator.svelte";


export { Root, Trigger, Content, Item, Separator };

// Re-export as namespace for convenience
export const ContextMenu = {
  Root,
  Trigger,
  Content,
  Item,
  Separator,
};
