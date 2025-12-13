// load-sidebar.js
fetch("sidebar.html")
  .then(response => response.text())
  .then(html => {
    document.getElementById("sidebar").innerHTML = html;
    loadRecentPosts();
  })
  .catch(err => console.error("Error loading sidebar:", err));

function loadRecentPosts() {
  const postList = document.getElementById("recent-posts");

  fetch("blog/blogs.json")
    .then(res => res.json())
    .then(posts => {
      // Sort posts by date descending (newest first)
      posts.sort((a, b) => new Date(b.date) - new Date(a.date));

      // Take top 3
      const recentPosts = posts.slice(0, 3);

      // Clear list just in case
      postList.innerHTML = "";

      recentPosts.forEach(post => {
        const li = document.createElement("li");
        // Link to post.html with just the filename (no blog/ prefix)
        li.innerHTML = `<a href="blog/post.html?file=${encodeURIComponent(post.file)}" target="_blank">${post.title}</a>`;
        postList.appendChild(li);
      });
    })
    .catch(error => {
      postList.innerHTML = `<li>Error loading posts: ${error.message}</li>`;
    });
}
