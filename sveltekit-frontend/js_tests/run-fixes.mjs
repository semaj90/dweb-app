console.log("🚀 Starting automatic error fixes for npm run check...\n");

import("./fix-all-svelte-errors.mjs")
  .then(() => {
    console.log("\n✅ Fix process completed successfully!");
  })
  .catch((error) => {
    console.error("\n❌ Error during fix process:", error);
    process.exit(1);
  });
