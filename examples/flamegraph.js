import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const BAR_HEIGHT = 22;
const BAR_GAP = 1;
const ROW_HEIGHT = BAR_HEIGHT + BAR_GAP;
const LABEL_PADDING = 6;
const MIN_BAR_PX = 1;
const MIN_LABEL_PX = 14;
const TOOLTIP_OFFSET = 14;
const ZOOM_DURATION = 480;
const FONT = `11px Inter, Roboto, "Helvetica Neue", "Arial Nova", "Nimbus Sans", Arial, sans-serif`;
const FONT_FOCUSED = `bold ${FONT}`;

function assignLayout(node, x0, x1) {
  node.x0 = x0;
  node.x1 = x1;
  if (!node.children) {
    return;
  }
  node.children.reduce((cursor, child) => {
    const childWidth = ((x1 - x0) * child.data.visits) / node.data.visits;
    assignLayout(child, cursor, cursor + childWidth);
    return cursor + childWidth;
  }, x0);
}

function drawSubtree(ctx, node, winX0, winX1, canvasWidth, scale, focusNode) {
  if (node.x1 <= winX0 || node.x0 >= winX1) {
    return;
  }

  const barWidth = (node.x1 - node.x0) * scale - BAR_GAP;
  if (barWidth < MIN_BAR_PX) {
    return;
  }

  const barX = (node.x0 - winX0) * scale;
  const barY = node.depth * ROW_HEIGHT;

  ctx.fillStyle = node.color;
  ctx.fillRect(barX, barY, barWidth, BAR_HEIGHT);

  if (barWidth >= MIN_LABEL_PX) {
    const labelX = Math.max(0, barX);
    const labelWidth = Math.min(canvasWidth, barX + barWidth) - labelX;
    if (labelWidth >= MIN_LABEL_PX) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(labelX, barY, labelWidth, BAR_HEIGHT);
      ctx.clip();
      ctx.font = node === focusNode ? FONT_FOCUSED : FONT;
      ctx.fillStyle = "#2e3440";
      ctx.fillText(node.data.label, labelX + LABEL_PADDING, barY + BAR_HEIGHT / 2);
      ctx.restore();
    }
  }

  if (!node.children) {
    return;
  }
  for (const child of node.children) {
    drawSubtree(ctx, child, winX0, winX1, canvasWidth, scale, focusNode);
  }
}

function hitTest(nodesByDepth, pointerX, pointerY, winX0, scale) {
  const row = nodesByDepth.get(Math.floor(pointerY / ROW_HEIGHT));
  if (!row) {
    return null;
  }

  const normX = winX0 + pointerX / scale;
  return (
    row.find(
      (node) =>
        node.x0 <= normX &&
        normX < node.x1 &&
        (node.x1 - node.x0) * scale - BAR_GAP >= MIN_BAR_PX,
    ) ?? null
  );
}

document.addEventListener("DOMContentLoaded", () => {
  const raw = JSON.parse(document.getElementById("flamegraph-data").textContent);
  const root = d3.hierarchy(raw, (d) => d.children);

  assignLayout(root, 0, 1);

  const allNodes = root.descendants();
  const canvasHeight = (d3.max(allNodes, (d) => d.depth) + 1) * ROW_HEIGHT;

  for (const node of allNodes) {
    node.color = d3.interpolateSpectral(1 - node.data.win_rate);
  }

  const nodesByDepth = d3.group(allNodes, (d) => d.depth);

  const container = d3.select("#flamegraph-container");
  const tooltip = d3.select("body").append("div").attr("id", "fg-tooltip");

  let canvasWidth = container.node().clientWidth;
  const dpr = window.devicePixelRatio || 1;

  const canvasSel = container
    .append("canvas")
    .attr("height", canvasHeight * dpr)
    .style("height", `${canvasHeight}px`);

  const ctx = canvasSel.node().getContext("2d");

  function resizeCanvas(width) {
    canvasSel.attr("width", width * dpr).style("width", `${width}px`);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resizeCanvas(canvasWidth);

  const xScale = d3.scaleLinear().domain([0, 1]).range([0, canvasWidth]);
  let focusNode = root;

  function draw() {
    const [winX0, winX1] = xScale.domain();
    const scale = canvasWidth / (winX1 - winX0);

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.font = FONT;
    ctx.textBaseline = "middle";

    drawSubtree(ctx, root, winX0, winX1, canvasWidth, scale, focusNode);
  }

  function zoomTo(node, animate) {
    const [winX0, winX1] = xScale.domain();
    focusNode = node;

    if (animate) {
      const interpX0 = d3.interpolateNumber(winX0, node.x0);
      const interpX1 = d3.interpolateNumber(winX1, node.x1);
      d3.transition()
        .duration(ZOOM_DURATION)
        .ease(d3.easeCubicInOut)
        .tween("zoom", () => (t) => {
          xScale.domain([interpX0(t), interpX1(t)]);
          draw();
        })
        .on("end", () => {
          xScale.domain([node.x0, node.x1]);
          draw();
        });
    } else {
      xScale.domain([node.x0, node.x1]);
      draw();
    }
  }

  function nodeAt(pointerX, pointerY) {
    const [winX0, winX1] = xScale.domain();
    return hitTest(
      nodesByDepth,
      pointerX,
      pointerY,
      winX0,
      canvasWidth / (winX1 - winX0),
    );
  }

  canvasSel
    .on("click", (event) => {
      const [pointerX, pointerY] = d3.pointer(event);
      const hit = nodeAt(pointerX, pointerY);
      if (hit) {
        zoomTo(hit === focusNode ? (hit.parent ?? root) : hit, true);
      } else if (focusNode !== root) {
        zoomTo(root, true);
      }
    })
    .on("mousemove", (event) => {
      const [pointerX, pointerY] = d3.pointer(event);
      const hit = nodeAt(pointerX, pointerY);
      tooltip.classed("visible", !!hit);
      container.classed("hovered", !!hit);
      if (hit) {
        tooltip
          .text(hit.data.title)
          .style("left", `${event.clientX + TOOLTIP_OFFSET}px`)
          .style("top", `${event.clientY + TOOLTIP_OFFSET}px`);
      }
    })
    .on("mouseleave", () => {
      tooltip.classed("visible", false);
      container.classed("hovered", false);
    });

  zoomTo(root, false);

  new ResizeObserver((entries) => {
    const newWidth = Math.round(entries[0].contentRect.width);
    if (Math.abs(newWidth - canvasWidth) < 1) {
      return;
    }
    canvasWidth = newWidth;
    xScale.range([0, canvasWidth]);
    resizeCanvas(canvasWidth);
    draw();
  }).observe(container.node());
});
