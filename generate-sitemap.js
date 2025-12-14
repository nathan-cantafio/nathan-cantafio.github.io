// generate-sitemap.js

const fs = require("fs");
const path = require("path");

// -------- CONFIG --------
const SITE_URL = "https://cantafionathan.github.io";
const BLOGS_JSON = path.join(__dirname, "blog", "blogs.json");
const OUTPUT = path.join(__dirname, "sitemap.xml");
// ------------------------

const posts = JSON.parse(fs.readFileSync(BLOGS_JSON, "utf8"));

const staticPages = [
  "/",
  "/all-posts.html",
  "/notes.html",
  "/cv/",
];

function url(loc, lastmod = null) {
  return `
  <url>
    <loc>${SITE_URL}${loc}</loc>
    ${lastmod ? `<lastmod>${lastmod}</lastmod>` : ""}
  </url>`;
}

let xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">`;

// Static pages
staticPages.forEach(page => {
  xml += url(page);
});

// Blog posts
posts.forEach(post => {
  const encoded = encodeURIComponent(post.file);
  const loc = `/blog/post.html?file=${encoded}`;
  const lastmod = post.date || null; // YYYY-MM-DD works perfectly
  xml += url(loc, lastmod);
});

xml += `
</urlset>
`;

fs.writeFileSync(OUTPUT, xml.trim() + "\n");

console.log(`✅ sitemap.xml generated with ${posts.length} posts`);
