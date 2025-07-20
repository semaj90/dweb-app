console.log("Test script is running");
console.log("Current directory:", process.cwd());
console.log("Node version:", process.version);

const { execSync } = require("child_process");

try {
  console.log("Running npm run check...");
  const result = execSync("npm run check", {
    encoding: "utf8",
    stdio: "pipe",
  });
  console.log("Success! Output:", result);
} catch (error) {
  console.log("Error occurred:", error.message);
  console.log("Error stdout:", error.stdout);
  console.log("Error stderr:", error.stderr);
}
