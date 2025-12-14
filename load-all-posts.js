// load-all-posts.js
fetch("blog/blogs.json")
  .then(res => res.json())
  .then(posts => {
    const postList = document.getElementById("all-posts-list");

    posts.sort((a, b) => new Date(b.date) - new Date(a.date));
    postList.innerHTML = "";

    posts.forEach(post => {
      const li = document.createElement("li");
    
      li.innerHTML = `
        <a href="blog/post.html?file=${encodeURIComponent(post.file)}">
          ${post.title}
        </a>
        <div class="post-date">${new Date(post.date).toLocaleDateString()}</div>
        <div class="post-description">${post.description}</div>
      `;
    
      postList.appendChild(li);
    });
    
  })
  .catch(error => {
    document.getElementById("all-posts-list").innerHTML = `<li>Error loading posts: ${error.message}</li>`;
  });
