// load-post.js

// -----------------------------
// Load header (optional)
// -----------------------------
fetch("../header.html")
  .then((response) => response.text())
  .then((data) => {
    document.getElementById("header").innerHTML = data;
  });


// -----------------------------
// Simple front-matter parser
// -----------------------------
function parseFrontMatter(mdText) {
  // Match YAML front matter at the top enclosed by ---
  const match = /^---\n([\s\S]+?)\n---/.exec(mdText);
  if (!match) return { attributes: {}, body: mdText };

  const yaml = match[1];
  const body = mdText.slice(match[0].length);

  // Parse simple key: value pairs (no nested structures)
  const attributes = {};
  yaml.split("\n").forEach(line => {
    const [key, ...rest] = line.split(":");
    if (!key) return;
    attributes[key.trim()] = rest.join(":").trim();
  });

  return { attributes, body };
}


// -----------------------------
// Utility: escape HTML
// -----------------------------
function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


// -----------------------------
// Get ?file= from URL
// -----------------------------
function getMarkdownFile() {
  const params = new URLSearchParams(window.location.search);
  return params.get("file");
}


// -----------------------------
// Main loader
// -----------------------------
async function loadPost() {
  const file = getMarkdownFile();

  if (!file) {
    document.getElementById("post-content").innerHTML =
      "<p><em>No post specified.</em></p>";
    document.title = "Post not found";
    return;
  }

  try {
    const response = await fetch(file);
    if (!response.ok) throw new Error("Failed to load markdown file.");

    const mdText = await response.text();

    const parsed = parseFrontMatter(mdText);
    const meta = parsed.attributes;
    const content = parsed.body;

    // -----------------------------
    // Sanitize metadata
    // -----------------------------
    const safeTitle = meta.title ? escapeHTML(meta.title) : "Untitled";
    const safeDate = meta.date ? escapeHTML(meta.date) : "";
    const safeDescription = meta.description
      ? escapeHTML(meta.description)
      : "";

    // -----------------------------
    // SEO: page title
    // -----------------------------
    document.title = `${safeTitle} – Nathan Cantafio`;

    // -----------------------------
    // SEO: meta description
    // -----------------------------
    if (safeDescription) {
      let metaDesc = document.querySelector('meta[name="description"]');
      if (!metaDesc) {
        metaDesc = document.createElement("meta");
        metaDesc.name = "description";
        document.head.appendChild(metaDesc);
      }
      metaDesc.content = safeDescription;
    }

    // -----------------------------
    // Render metadata
    // -----------------------------
    document.getElementById("post-meta").innerHTML = `
      <h1>${safeTitle}</h1>
      <p><em>${safeDate}</em></p>
      <p>${safeDescription}</p>
      <hr>
    `;

    // -----------------------------
    // Render markdown content
    // -----------------------------
    document.getElementById("post-content").innerHTML =
      marked.parse(content);

  } catch (err) {
    document.getElementById("post-content").innerHTML =
      `<p><em>Error loading post:</em> ${escapeHTML(err.message)}</p>`;
    document.title = "Error loading post";
  }

  // -----------------------------
  // Typeset math
  // -----------------------------
  if (window.MathJax) {
    MathJax.typesetPromise();
  }
}


// -----------------------------
// DOM ready
// -----------------------------
document.addEventListener("DOMContentLoaded", loadPost);
