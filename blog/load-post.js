// load-post.js

// Load header (optional)
fetch("../header.html")
  .then((response) => response.text())
  .then((data) => {
    document.getElementById("header").innerHTML = data;
  });


function parseFrontMatter(mdText) {
  // Match YAML front matter at the top enclosed by ---
  const match = /^---\n([\s\S]+?)\n---/.exec(mdText);
  if (!match) return { attributes: {}, body: mdText };

  const yaml = match[1];
  const body = mdText.slice(match[0].length);

  // Parse simple key: value pairs (no nested structures)
  const attributes = {};
  yaml.split('\n').forEach(line => {
    const [key, ...rest] = line.split(':');
    if (!key) return;
    attributes[key.trim()] = rest.join(':').trim();
  });

  return { attributes, body };
}


// Function to get ?file= parameter from URL
function getMarkdownFile() {
  const params = new URLSearchParams(window.location.search);
  return params.get("file");
}

async function loadPost() {
  const file = getMarkdownFile();
  if (!file) {
    document.getElementById("post-content").innerHTML = "<p><em>No post specified.</em></p>";
    return;
  }

  try {
    const response = await fetch(file);
    if (!response.ok) throw new Error("Failed to load markdown file.");

    const mdText = await response.text();

	const parsed = parseFrontMatter(mdText);
	const meta = parsed.attributes;
	const content = parsed.body;


    // Render metadata separately (optional)
    document.getElementById("post-meta").innerHTML = `
      <h1>${meta.title || "Untitled"}</h1>
      <p><em>${meta.date || ""}</em></p>
      <p>${meta.description || ""}</p>
      <hr>
    `;

    // Render markdown content (only the content, no front matter)
    document.getElementById("post-content").innerHTML = marked.parse(content);
  } catch (err) {
    document.getElementById("post-content").innerHTML = `<p><em>Error loading post:</em> ${err.message}</p>`;
  }

  // Typeset math
  if (window.MathJax) {
    MathJax.typesetPromise();
  }
}

// Make sure to wait for DOM ready
document.addEventListener("DOMContentLoaded", loadPost);
