document.addEventListener("DOMContentLoaded", () => {
    fetch("/header.html?v=" + Date.now())
        .then(response => response.text())
        .then(html => {
            document.getElementById("header").innerHTML = html;

            const path = window.location.pathname;
            let activePage = "home";
            if (path.includes("all-posts") || path.includes("/blog/")) activePage = "blog";
            else if (path.includes("notes")) activePage = "notes";

            document.querySelectorAll(".header-nav a[data-page]").forEach(link => {
                if (link.dataset.page === activePage) link.classList.add("active");
            });
        })
        .catch(err => console.error("Failed to load header:", err));
});
