
/**
 * Draggable action for making elements draggable on the canvas
 */
export function draggable(
  node: HTMLElement,
  options: {
    onDrag?: (x: number, y: number) => void;
    onDragStart?: () => void;
    onDragEnd?: () => void;
    handle?: string; // CSS selector for drag handle
  } = {},
) {
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let initialX = 0;
  let initialY = 0;

  function handleMouseDown(event: MouseEvent) {
    // Check if we're dragging from the handle
    if (options.handle) {
      const handle = node.querySelector(options.handle);
      if (!handle || !handle.contains(event.target as Node)) {
        return;
      }
    }
    isDragging = true;
    startX = event.clientX;
    startY = event.clientY;

    // Get current position
    const rect = node.getBoundingClientRect();
    initialX = rect.left;
    initialY = rect.top;

    // Add global event listeners
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    // Prevent text selection
    event.preventDefault();

    if (options.onDragStart) {
      options.onDragStart();
    }
    // Add dragging class
    node.classList.add("dragging");
  }
  function handleMouseMove(event: MouseEvent) {
    if (!isDragging) return;

    const deltaX = event.clientX - startX;
    const deltaY = event.clientY - startY;

    const newX = initialX + deltaX;
    const newY = initialY + deltaY;

    // Constrain to viewport bounds
    const maxX = window.innerWidth - node.offsetWidth;
    const maxY = window.innerHeight - node.offsetHeight;

    const constrainedX = Math.max(0, Math.min(maxX, newX));
    const constrainedY = Math.max(0, Math.min(maxY, newY));

    if (options.onDrag) {
      options.onDrag(constrainedX, constrainedY);
    }
  }
  function handleMouseUp() {
    if (!isDragging) return;

    isDragging = false;

    // Remove global event listeners
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);

    if (options.onDragEnd) {
      options.onDragEnd();
    }
    // Remove dragging class
    node.classList.remove("dragging");
  }
  // Add mouse down listener to the node
  node.addEventListener("mousedown", handleMouseDown);

  // Cleanup function
  return {
    destroy() {
      node.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    },

    update(newOptions: typeof options) {
      options = { ...options, ...newOptions };
    },
  };
}
