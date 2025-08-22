import Fuse from "fuse.js";
import Loki from "lokijs";


// 1. Initialize LokiJS Database
const db = new Loki("fuzzy.db");
const items = db.addCollection("items");

// 2. Sample Data (replace with your actual data)
items.insert([
  { id: 1, title: "State v. John Doe", type: "case" },
  { id: 2, title: "Evidence Locker #123", type: "evidence" },
  { id: 3, title: "Witness Statement - Jane Smith", type: "document" },
]);

// 3. Configure Fuse.js
const options = {
  includeScore: true,
  keys: ["title", "type"],
};

const fuse = new Fuse(items.data, options);

// 4. Fuzzy Search Function
export function fuzzySearch(query) {
  return fuse.search(query);
}
