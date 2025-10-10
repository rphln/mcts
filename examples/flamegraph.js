document.addEventListener("DOMContentLoaded", () => {
  document.addEventListener("click", (e) => {
    let bar = e.target.closest(".bar");
    let node = bar?.closest(".node");

    if (!node) {
      return;
    }

    requestAnimationFrame(() => {
      for (const el of node.querySelectorAll(".focused, .collapsed")) {
        el.classList.remove("focused", "collapsed");
      }

      while (node) {
        if (node.classList.contains("focused")) {
          break;
        }

        node.classList.add("focused");

        let prevSibling = node.previousElementSibling;
        while (prevSibling) {
          if (prevSibling.classList.contains("node")) {
            prevSibling.classList.add("collapsed");
          }
          prevSibling = prevSibling.previousElementSibling;
        }

        let nextSibling = node.nextElementSibling;
        while (nextSibling) {
          if (nextSibling.classList.contains("node")) {
            nextSibling.classList.add("collapsed");
          }
          nextSibling = nextSibling.nextElementSibling;
        }

        node = node.parentElement?.closest(".node");
      }
    });
  });
});
