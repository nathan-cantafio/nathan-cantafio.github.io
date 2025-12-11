document.addEventListener("DOMContentLoaded", () => {
    fetch("header.html?cache_bust=" + Date.now())
        .then(response => response.text())
        .then(html => {
            document.getElementById("header").innerHTML = html;
        })
        .catch(err => console.error("Failed to load header:", err));
});
