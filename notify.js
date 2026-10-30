// notify.js: email Kit subscribers about the newest post in feed.xml.
//
//   node notify.js          creates a draft broadcast in Kit to review and send
//   node notify.js --send   schedules it to send in 5 minutes (cancel in Kit if needed)
//
// Needs your Kit v4 API key in the KIT_API_KEY environment variable
// (Kit → Settings → Developer). Keep it out of this repo, e.g. in ~/.zshrc:
//   export KIT_API_KEY="..."
//
// Run it after pushing live, so the link in the email works. Needs Node 18+.

const fs = require("fs");
const path = require("path");

const key = process.env.KIT_API_KEY;
if (!key) {
  console.error("notify.js: set KIT_API_KEY first (Kit → Settings → Developer).");
  process.exit(1);
}
const send = process.argv.includes("--send");

// The newest post is the first <item>; build.js writes the feed newest first.
const xml = fs.readFileSync(path.join(__dirname, "feed.xml"), "utf8");
const item = xml.match(/<item>([\s\S]*?)<\/item>/);
if (!item) {
  console.error("notify.js: no posts in feed.xml.");
  process.exit(1);
}
const get = (tag) => item[1].match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`))[1];
const unescape = (s) =>
  s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
   .replace(/&#39;/g, "'").replace(/&amp;/g, "&");

// Feed values are XML-escaped, which is also safe to put straight into HTML.
const title = get("title");
const link = get("link");
const description = get("description");

(async () => {
  const page = await fetch(link);
  if (!page.ok) {
    console.error(`notify.js: ${link} isn't live yet (HTTP ${page.status}). Push live first.`);
    process.exit(1);
  }

  const broadcast = {
    subject: `New post: ${unescape(title)}`,
    preview_text: unescape(description),
    content:
      `<p>I've published a new post:</p>` +
      `<h2><a href="${link}">${title}</a></h2>` +
      `<p>${description}</p>` +
      `<p><a href="${link}">Read it on the blog →</a></p>`,
    description: `Blog post: ${unescape(title)}`, // internal note; only you see it
    public: false,
    send_at: send ? new Date(Date.now() + 5 * 60 * 1000).toISOString() : null,
  };

  const res = await fetch("https://api.kit.com/v4/broadcasts", {
    method: "POST",
    headers: {
      "X-Kit-Api-Key": key,
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(broadcast),
  });
  const out = await res.json().catch(() => ({}));
  if (!res.ok) {
    console.error(`notify.js: Kit returned HTTP ${res.status}:`, JSON.stringify(out));
    process.exit(1);
  }

  console.log(
    send
      ? `Scheduled "${unescape(title)}" for ${broadcast.send_at}. Cancel in Kit → Broadcasts if needed.`
      : `Draft created for "${unescape(title)}". Review and send it in Kit → Broadcasts.`
  );
})();
