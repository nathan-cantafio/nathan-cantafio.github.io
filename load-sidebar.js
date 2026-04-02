document.addEventListener("DOMContentLoaded", () => {
    fetch("/sidebar.html?v=" + Date.now())
        .then(res => res.text())
        .then(html => {
            document.getElementById("sidebar").innerHTML = html;
            return fetch("/blog/blogs.json");
        })
        .then(res => res.json())
        .then(posts => {
            posts.sort((a, b) => new Date(b.date) - new Date(a.date));
            const recent = posts.slice(0, 3);

            // Sidebar recent posts
            const sidebarList = document.getElementById("recent-posts");
            if (sidebarList) {
                sidebarList.innerHTML = recent.map(p =>
                    `<li><a href="/blog/post.html?file=${encodeURIComponent(p.file)}">${p.title}</a></li>`
                ).join("");
            }

            // Homepage recent writing section
            const mainList = document.getElementById("recent-posts-main");
            if (mainList) {
                mainList.innerHTML = recent.map(p => {
                    const date = new Date(p.date + "T00:00:00").toLocaleDateString("en-US", { month: "short", year: "numeric" });
                    return `<li>
                        <a href="/blog/post.html?file=${encodeURIComponent(p.file)}">${p.title}</a>
                        <span class="post-date">${date}</span>
                    </li>`;
                }).join("");
            }
        })
        .catch(err => console.error("Failed to load sidebar:", err));
});
