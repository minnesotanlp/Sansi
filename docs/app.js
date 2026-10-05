/* SanSi project page: charts and the item explorer. No dependencies; all numbers come from data/site.json and
   data/items.json (written by build_data.py from the result files of the paper). */
(() => {
"use strict";

const NS = "http://www.w3.org/2000/svg";
const COL = { sansi: "#0b5cad", dark: "#05295c", smol: "#d9480f", qwen: "#1b8a4c", kev: "#a8327e", jev: "#586172", gray: "#808a9b", grayLight: "#c7cdd8", surface: "#ffffff", wash: "#f1f4f9", ink2: "#454f63" };
const $ = (sel, root = document) => root.querySelector(sel);

function setAttrs(node, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) node.setAttribute(k, v);
  return node;
}
function h(tag, attrs, parent, text) {
  const node = setAttrs(document.createElement(tag), attrs);
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}
function s(tag, attrs, parent, text) {
  const node = setAttrs(document.createElementNS(NS, tag), attrs);
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const signed = (v) => (v > -0.05 ? "+" : "−") + Math.abs(v).toFixed(1);
const iv = (d) => "[" + f1(Math.abs(d[1]) < 0.05 ? 0 : d[1]) + ", " + f1(Math.abs(d[2]) < 0.05 ? 0 : d[2]) + "]";
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const logScale = (d0, d1, r0, r1) => (v) => r0 + ((Math.log(v) - Math.log(d0)) / (Math.log(d1) - Math.log(d0))) * (r1 - r0);

/* Round tick positions that cover [lo, hi] in about n steps. */
function niceTicks(lo, hi, n = 5) {
  const raw = (hi - lo) / n, pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * pow).find((k) => k >= raw);
  const a = Math.floor(lo / step + 1e-9) * step, b = Math.ceil(hi / step - 1e-9) * step, ticks = [];
  for (let v = a; v <= b + step * 1e-6; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return { domain: [a, b], ticks, step };
}
/* Tick labels: as many decimals as the ticks need, the same for all of them. */
function tickFmt(ticks) {
  const d = Math.max(...ticks.map((t) => ((String(t).split(".")[1]) || "").length));
  return (v) => v.toFixed(d);
}

/* ------------------------------------------------------------------ tooltip */
const tipEl = $("#tip");
function tipFill(title, rows) {
  tipEl.textContent = "";
  if (title) h("div", { class: "t" }, tipEl, title);
  for (const r of rows) {
    const row = h("div", { class: "r" }, tipEl), name = h("span", { class: "n" }, row);
    if (r.color) {
      const key = h("i", { class: "key" }, name);
      key.style.borderTopColor = r.color;
      if (r.dash) key.style.borderTopStyle = "dashed";
    }
    name.appendChild(document.createTextNode(r.label));
    h("b", {}, row, r.value);
  }
  tipEl.classList.add("on");
}
function tipAt(x, y) {
  const w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
  let left = x + 14, top = y - hh - 12;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  if (left < 8) left = 8;
  if (top < 8) top = y + 18;
  tipEl.style.left = left + "px";
  tipEl.style.top = top + "px";
}
function tipHide() { tipEl.classList.remove("on"); }
/* A mark with its own tooltip: pointer and keyboard focus show the same rows. */
function hover(node, title, rows, focusable) {
  node.addEventListener("pointermove", (e) => { tipFill(title, rows); tipAt(e.clientX, e.clientY); });
  node.addEventListener("pointerleave", tipHide);
  if (!focusable) return;
  node.setAttribute("tabindex", "0");
  node.setAttribute("aria-label", title + ": " + rows.map((r) => r.label + " " + r.value).join(", "));
  node.addEventListener("focus", () => { const b = node.getBoundingClientRect(); tipFill(title, rows); tipAt(b.left + b.width / 2, b.top + 20); });
  node.addEventListener("blur", tipHide);
}

/* ------------------------------------------------------------------ chart pieces */
function legend(host, items) {
  const box = h("div", { class: "legend" }, host);
  for (const it of items) {
    const k = h("span", { class: "k" }, box), g = s("svg", { width: 26, height: 12, viewBox: "0 0 26 12", "aria-hidden": "true" }, k);
    if (it.kind === "rect") s("rect", { x: 7, y: 1, width: 12, height: 10, rx: 2, fill: it.color }, g);
    else {
      if (it.kind !== "mark") s("line", { x1: 1, y1: 6, x2: 25, y2: 6, stroke: it.color, "stroke-width": it.width || 2, "stroke-dasharray": it.dash || null }, g);
      if (it.marker === "dot") s("circle", { cx: 13, cy: 6, r: 4, fill: it.hollow ? COL.surface : it.color, stroke: it.color, "stroke-width": 1.5 }, g);
      if (it.marker === "square") s("rect", { x: 9, y: 2, width: 8, height: 8, fill: it.hollow ? COL.surface : it.color, stroke: it.color, "stroke-width": 1.5 }, g);
      if (it.marker === "tick") s("line", { x1: 13, y1: 0, x2: 13, y2: 12, stroke: it.color, "stroke-width": 3 }, g);
      if (it.marker === "triangle") s("path", { d: "M13 0.5 L19 11 L7 11 Z", fill: it.hollow ? COL.surface : it.color, stroke: it.color, "stroke-width": 1.5 }, g);
    }
    k.appendChild(document.createTextNode(it.label));
  }
  return box;
}
function table(host, head, rows) {
  const d = h("details", { class: "tbl" }, host);
  h("summary", {}, d, "Show the numbers");
  const t = h("table", {}, d), hr = h("tr", {}, h("thead", {}, t)), tb = h("tbody", {}, t);
  head.forEach((x) => h("th", { scope: "col" }, hr, x));
  rows.forEach((r) => { const tr = h("tr", {}, tb); r.forEach((c) => h("td", {}, tr, c === null || c === undefined ? "–" : String(c))); });
}
function base(host, W, H, m, aria) {
  const box = h("div", { class: "chart", role: "img", "aria-label": aria }, host);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` }, box);
  return { box, svg, W, H, x0: m.l, x1: W - m.r, y0: m.t, y1: H - m.b };
}
function yAxis(c, y, ticks, fmt, title) {
  for (const t of ticks) {
    s("line", { class: "grid", x1: c.x0, x2: c.x1, y1: y(t), y2: y(t) }, c.svg);
    s("text", { x: c.x0 - 8, y: y(t) + 4, "text-anchor": "end" }, c.svg, fmt(t));
  }
  if (title) s("text", { class: "lab", x: 2, y: c.y0 - 12 }, c.svg, title);
}
function xAxis(c, x, ticks, fmt, title) {
  s("line", { class: "axis", x1: c.x0, x2: c.x1, y1: c.y1, y2: c.y1 }, c.svg);
  for (const t of ticks) s("text", { x: x(t), y: c.y1 + 17, "text-anchor": "middle" }, c.svg, fmt(t));
  if (title) s("text", { class: "lab", x: (c.x0 + c.x1) / 2, y: c.y1 + 36, "text-anchor": "middle" }, c.svg, title);
}
function dot(parent, cx, cy, color, opts = {}) {
  return s("circle", { cx, cy, r: opts.r || 4.5, fill: opts.hollow ? COL.surface : color, stroke: opts.hollow ? color : COL.surface, "stroke-width": 2, opacity: opts.opacity || null }, parent);
}
function square(parent, cx, cy, color, opts = {}) {
  const a = opts.size || 9;
  return s("rect", { x: cx - a / 2, y: cy - a / 2, width: a, height: a, fill: opts.hollow ? COL.surface : color, stroke: opts.hollow ? color : COL.surface, "stroke-width": 2 }, parent);
}
function triangle(parent, cx, cy, color, opts = {}) {
  const a = opts.size || 6.5;
  return s("path", { d: `M${cx} ${cy - a} L${cx + a} ${cy + a * 0.8} L${cx - a} ${cy + a * 0.8} Z`, fill: opts.hollow ? COL.surface : color, stroke: opts.hollow ? color : COL.surface, "stroke-width": 2, "stroke-linejoin": "round" }, parent);
}
/* Labels at the right edge, moved apart when they would overlap, each tied to its mark by a short leader. */
function endLabels(c, labels) {
  labels.sort((a, b) => a.y - b.y);
  const gap = 15;
  for (let i = 1; i < labels.length; i++) if (labels[i].ly - labels[i - 1].ly < gap) labels[i].ly = labels[i - 1].ly + gap;
  const over = labels.length ? labels[labels.length - 1].ly - (c.y1 + 4) : 0;
  if (over > 0) labels.forEach((l) => { l.ly -= over; });
  for (const l of labels) {
    s("line", { x1: c.x1 + 3, y1: l.y, x2: c.x1 + 14, y2: l.ly, stroke: l.color, "stroke-width": 1.5, "stroke-dasharray": l.dash || null }, c.svg);
    s("text", { class: "lab", x: c.x1 + 18, y: l.ly + 4 }, c.svg, l.text);
  }
}

/* A line chart over a shared x: series with optional bands, horizontal reference lines, a crosshair with one tooltip. */
function lineChart(host, cfg) {
  host.textContent = "";
  const items = [];
  cfg.series.forEach((se) => items.push({ color: se.color, dash: se.dash, marker: se.markers === false ? null : "dot", hollow: se.hollow, label: se.name, width: se.width }));
  (cfg.refs || []).forEach((r) => items.push({ color: r.color, dash: r.dash, label: r.name, width: 1.5 }));
  if (cfg.legendItems) legend(host, cfg.legendItems);
  else if (items.length > 1 && cfg.legend !== false) legend(host, items);
  const W = cfg.W || 640, H = cfg.H || 330, m = Object.assign({ t: 30, r: cfg.endLabels ? 128 : 16, b: cfg.xTitle ? 48 : 30, l: 46 }, cfg.m || {});
  const c = base(host, W, H, m, cfg.aria);
  const xs = cfg.xs, x = (cfg.xLog ? logScale : lin)(cfg.xDomain[0], cfg.xDomain[1], c.x0, c.x1), y = lin(cfg.yDomain[0], cfg.yDomain[1], c.y1, c.y0);
  for (const r of cfg.regions || []) {
    s("rect", { x: x(r.from), y: c.y0, width: x(r.to) - x(r.from), height: c.y1 - c.y0, fill: COL.wash }, c.svg);
    s("text", { x: (x(r.from) + x(r.to)) / 2, y: c.y0 + 14, "text-anchor": "middle" }, c.svg, r.label);
  }
  yAxis(c, y, cfg.yTicks, tickFmt(cfg.yTicks), cfg.yTitle);
  xAxis(c, x, cfg.xTicks || xs, cfg.xFmt || String, cfg.xTitle);
  const labels = [];
  for (const r of cfg.refs || []) {
    s("line", { x1: c.x0, x2: c.x1, y1: y(r.value), y2: y(r.value), stroke: r.color, "stroke-width": 1.5, "stroke-dasharray": r.dash || null }, c.svg);
    if (cfg.endLabels) labels.push({ y: y(r.value), ly: y(r.value), text: r.short || r.name, color: r.color, dash: r.dash });
  }
  if (cfg.mark !== undefined) s("line", { class: "cross", x1: x(cfg.mark), x2: x(cfg.mark), y1: c.y0, y2: c.y1, opacity: 0.55 }, c.svg);
  for (const se of cfg.series) {
    const pts = xs.map((v, i) => (se.values[i] === null || se.values[i] === undefined ? null : [x(v), y(se.values[i]), i])).filter(Boolean);
    if (se.sd) {
      const up = pts.map(([px, , i]) => `${px},${y(se.values[i] + se.sd[i])}`), dn = pts.map(([px, , i]) => `${px},${y(se.values[i] - se.sd[i])}`).reverse();
      s("polygon", { points: up.concat(dn).join(" "), fill: se.color, opacity: 0.12 }, c.svg);
    }
    s("path", { d: pts.map(([px, py], i) => (i ? "L" : "M") + px + " " + py).join(" "), fill: "none", stroke: se.color, "stroke-width": se.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": se.dash || null }, c.svg);
    if (se.markers !== false) for (const [px, py, i] of pts) dot(c.svg, px, py, se.color, { hollow: se.hollow || (se.hollowFrom !== undefined && xs[i] > se.hollowFrom) });
    if (cfg.endLabels && pts.length) { const last = pts[pts.length - 1]; labels.push({ y: last[1], ly: last[1], text: se.short || se.name, color: se.color, dash: se.dash }); }
  }
  if (cfg.endLabels) endLabels(c, labels);
  /* crosshair: aim at an x, read every series */
  const cross = s("line", { class: "cross", y1: c.y0, y2: c.y1, visibility: "hidden" }, c.svg);
  const hit = s("rect", { x: c.x0, y: c.y0, width: c.x1 - c.x0, height: c.y1 - c.y0, fill: "transparent" }, c.svg);
  const rowsAt = (i) => cfg.series.filter((se) => se.values[i] !== null && se.values[i] !== undefined).map((se) => ({ color: se.color, dash: se.dash, label: se.name, value: cfg.yFmt(se.values[i]) + (se.sd ? " ± " + cfg.yFmt(se.sd[i]) : "") }))
    .concat((cfg.refs || []).map((r) => ({ color: r.color, dash: r.dash, label: r.name, value: cfg.yFmt(r.value) })));
  let cur = -1;
  const show = (i, cx, cy) => { cur = i; setAttrs(cross, { x1: x(xs[i]), x2: x(xs[i]), visibility: "visible" }); tipFill(cfg.tipTitle ? cfg.tipTitle(xs[i]) : String(xs[i]), rowsAt(i)); tipAt(cx, cy); };
  const off = () => { cross.setAttribute("visibility", "hidden"); tipHide(); };
  hit.addEventListener("pointermove", (e) => {
    const b = c.svg.getBoundingClientRect(), px = ((e.clientX - b.left) / b.width) * W;
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(x(v) - px) < Math.abs(x(xs[best]) - px)) best = i; });
    show(best, e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", off);
  c.box.setAttribute("tabindex", "0");
  c.box.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const i = Math.max(0, Math.min(xs.length - 1, cur + (e.key === "ArrowRight" ? 1 : -1))), b = c.svg.getBoundingClientRect();
    show(i, b.left + (x(xs[i]) / W) * b.width, b.top + (c.y0 / H) * b.height + 30);
  });
  c.box.addEventListener("blur", off);
  return c;
}

/* A column that grows from the baseline, rounded at its data end only. */
function column(parent, cx, w, yBase, yTip, color) {
  const up = yTip < yBase, r = Math.min(4, Math.abs(yBase - yTip)), x0 = cx - w / 2, x1 = cx + w / 2, k = up ? 1 : -1;
  const d = `M${x0} ${yBase} L${x0} ${yTip + k * r} Q${x0} ${yTip} ${x0 + r} ${yTip} L${x1 - r} ${yTip} Q${x1} ${yTip} ${x1} ${yTip + k * r} L${x1} ${yBase} Z`;
  return s("path", { d, fill: color }, parent);
}

/* ------------------------------------------------------------------ segmented controls and chips: buttons in front of a <select> */
function segment(sel, cls, shorts) {
  const box = h("div", { class: cls, role: "group", "aria-label": sel.getAttribute("aria-label") });
  sel.parentNode.insertBefore(box, sel.nextSibling);
  sel.hidden = true;
  const btns = Array.from(sel.options).map((o) => {
    const b = h("button", { type: "button", title: shorts && shorts[o.value] ? o.textContent : null }, box, shorts && shorts[o.value] ? shorts[o.value] : o.textContent);
    b.addEventListener("click", () => { sel.value = o.value; sel.dispatchEvent(new Event("change")); });
    return b;
  });
  const sync = () => btns.forEach((b, i) => b.setAttribute("aria-pressed", String(sel.options[i].value === sel.value)));
  sel.addEventListener("change", sync);
  sync();
}

/* ------------------------------------------------------------------ page chrome: navigation, reading progress, citation */
(function chrome() {
  const nav = $("#navbar"), toggle = $("#nav-toggle"), bar = $("#progress"), links = Array.from(document.querySelectorAll("#nav-links a"));
  const secs = links.map((a) => $(a.getAttribute("href")));
  toggle.addEventListener("click", () => toggle.setAttribute("aria-expanded", String(nav.classList.toggle("open"))));
  links.forEach((a) => a.addEventListener("click", () => { nav.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }));
  function onScroll() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = (max > 0 ? Math.min(1, window.scrollY / max) * 100 : 0) + "%";
    let on = -1;
    secs.forEach((sec, i) => { if (sec && sec.getBoundingClientRect().top <= 130) on = i; });
    links.forEach((a, i) => a.classList.toggle("on", i === on));
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  onScroll();
  const copy = $("#bib-copy");
  copy.addEventListener("click", () => {
    if (!navigator.clipboard || !navigator.clipboard.writeText) return;
    navigator.clipboard.writeText($("#bib").textContent).then(() => { copy.textContent = "Copied"; setTimeout(() => { copy.textContent = "Copy"; }, 1600); }, () => {});
  });
})();

/* ------------------------------------------------------------------ the page */
Promise.all([fetch("data/site.json").then((r) => r.json()), fetch("data/items.json").then((r) => r.json())]).then(([S, IT]) => {
  const LOOPS = [1, 2, 3, 4, 5, 6, 7, 8];
  const ref = (arm, group, key) => S.refs[arm][group][key][0];
  const holds = (acc) => { let k = 0; while (k < acc.length && acc[k] >= 75) k++; return k; };

  /* ---------------------------------------------------------------- the animated figure at the top: one item, loop by loop */
  (function teaser() {
    const fig = $("#teaser"), items = IT.items.filter((it) => it.paper);
    if (!fig || !items.length) return;
    $("#n-items").textContent = String(IT.items.length);
    const LETTER = "ABCDEFGH", WHAT = { fixed: "fixed by the loops", broken: "broken by the loops", always_right: "right at every loop", always_wrong: "wrong at every loop", back_and_forth: "moved back and forth by the loops" };
    const argmax = (p) => p.indexOf(Math.max(...p)), cut = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);

    /* the diagram: one stack of layers, a way back to its input, and a readout */
    const W = 360, H = 300, X = 150, TOP = 74, BOT = 236, J = 44, L = 50, B = 264, R = 14, RX = 236, RW = 92;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "One stack of 24 layers is applied eight times. After every loop a readout gives the probability of every option, and the hidden state returns to the input of the stack for the next loop." }, $("#tz-model"));
    const dIn = `M0 ${B} H${X - R} Q${X} ${B} ${X} ${B - R} V${BOT}`, dUp = `M${X} ${BOT} V${J}`, dRead = `M${X} ${J} H${RX}`, dOut = `M${RX + RW} ${J} H${W}`;
    const dBack = `M${X} ${J} H${L + R} Q${L} ${J} ${L} ${J + R} V${B - R} Q${L} ${B} ${L + R} ${B} H${X - R} Q${X} ${B} ${X} ${B - R} V${BOT}`;
    for (const d of [dIn, dBack, `M${X} ${TOP} V${J}`, dRead, dOut]) s("path", { class: "tz-wire", d }, svg);
    for (const d of [`M${X - 5} ${BOT + 10} L${X} ${BOT + 2} L${X + 5} ${BOT + 10} Z`, `M${RX - 9} ${J - 5} L${RX - 1} ${J} L${RX - 9} ${J + 5} Z`, `M${W - 9} ${J - 5} L${W - 1} ${J} L${W - 9} ${J + 5} Z`, `M${L - 5} 150 L${L} 158 L${L + 5} 150 Z`])
      s("path", { class: "tz-head-arrow", d }, svg);
    s("rect", { class: "tz-stack", x: X - 70, y: TOP, width: 140, height: BOT - TOP, rx: 14 }, svg);
    const layers = [0, 1, 2, 3, 4].map((k) => { const y = BOT - 30 - k * 30; return { cy: y + 9, node: s("rect", { class: "tz-layer", x: X - 54, y, width: 108, height: 18, rx: 9 }, svg) }; });
    const readout = s("rect", { class: "tz-readout", x: RX, y: J - 15, width: RW, height: 30, rx: 15 }, svg);
    const readoutText = s("text", { class: "lab", x: RX + RW / 2, y: J + 4, "text-anchor": "middle" }, svg, "readout 1");
    s("text", { x: (L + X) / 2, y: J - 9, "text-anchor": "middle" }, svg, "next loop");
    s("text", { x: X, y: B + 24, "text-anchor": "middle" }, svg, "24 layers, the same weights in every loop");
    const CX = (X + 70 + W) / 2;
    s("text", { class: "kick", x: CX, y: 128, "text-anchor": "middle" }, svg, "LOOP");
    const counter = s("text", { class: "big", x: CX, y: 176, "text-anchor": "middle" }, svg, "1");
    s("text", { x: CX, y: 198, "text-anchor": "middle" }, svg, "of 8");
    const [pIn, pUp, pBack, pRead, pOut] = [dIn, dUp, dBack, dRead, dOut].map((d) => s("path", { d, fill: "none", stroke: "none" }, svg));
    const main = [s("circle", { class: "tz-halo", r: 11, visibility: "hidden" }, svg), s("circle", { class: "tz-pulse", r: 5.5, visibility: "hidden" }, svg)];
    const read = [s("circle", { class: "tz-read", r: 4.5, visibility: "hidden" }, svg)];

    /* the panels */
    const steps = LOOPS.map((t) => { const c = h("div", { class: "tz-step" }, $("#tz-steps")); c.appendChild(document.createTextNode(String(t))); h("i", {}, c); return c; });
    const dots = items.map((it, i) => { const b = h("button", { type: "button", "aria-label": "Example " + (i + 1) + ": " + it.source, title: it.source }, $("#tz-dots")); b.addEventListener("click", () => go(i, true)); return b; });
    const toggle = $("#tz-toggle");
    let rows = [], cur = 0, t = 1, phase = "in", el = 0, answered = false, last = null, raf = 0, visible = true;
    let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function setItem(i) {
      cur = i;
      const it = items[i];
      for (const id of ["#tz-tags", "#tz-opts"]) $(id).textContent = "";
      for (const tag of [it.source, it.type_name]) h("span", { class: "tag" }, $("#tz-tags"), tag);
      $("#tz-state").textContent = it.state;
      $("#tz-question").textContent = it.question;
      $("#tz-what").textContent = "Shown now: an item of " + it.source + " that is " + WHAT[it.group] + ".";
      rows = it.options.map((o, k) => {
        const row = h("div", { class: "opt" + (k === it.gold ? " is-gold" : "") }, $("#tz-opts")), txt = h("div", { class: "txt" }, row);
        h("b", {}, txt, LETTER[k] + "  "); txt.appendChild(document.createTextNode(cut(o, 46)));
        if (k === it.gold) h("span", { class: "gold" }, txt, "correct");
        h("div", { class: "p" }, row); h("div", { class: "fill" }, h("div", { class: "track" }, row));
        return row;
      });
      dots.forEach((d, j) => d.setAttribute("aria-pressed", String(j === i)));
      steps.forEach((c) => { c.className = "tz-step"; c.lastChild.textContent = ""; });
      $("#tz-loop").textContent = "before loop 1";
      const vd = $("#tz-verdict"); vd.className = "verdict"; vd.textContent = "";
    }
    function setLoop(tt) { counter.textContent = String(tt); readoutText.textContent = "readout " + tt; }
    function showAnswer(tt) {
      const it = items[cur], p = it.p[tt - 1], top = argmax(p), ok = top === it.gold;
      $("#tz-loop").textContent = "after loop " + tt;
      p.forEach((v, k) => { rows[k].classList.toggle("is-top", k === top); $(".p", rows[k]).textContent = f2(v); $(".fill", rows[k]).style.width = v * 100 + "%"; });
      const vd = $("#tz-verdict"); vd.textContent = ""; vd.className = "verdict" + (ok ? " ok" : "");
      h("span", { class: "mark" }, vd, ok ? "✓" : "✗");
      h("b", {}, vd, " Answers " + LETTER[top] + " (" + f2(p[top]) + ")");
      vd.appendChild(document.createTextNode(ok ? ": correct." : ": wrong; the correct option is " + LETTER[it.gold] + "."));
      steps.forEach((c, j) => c.classList.toggle("now", j === tt - 1));
      steps[tt - 1].classList.add(ok ? "right" : "wrong");
      steps[tt - 1].lastChild.textContent = LETTER[top];
    }

    /* the motion: enter, up through the layers, then to the readout and back to the input */
    const DUR = { in: 520, up: 560, split: 680, read: 300, last: 560, hold: 3000, fade: 320 };
    const dur = () => (phase === "split" && t === 8 ? DUR.last : DUR[phase]);
    const at = (path, u) => path.getPointAtLength(Math.max(0, Math.min(1, u)) * path.getTotalLength());
    const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
    const put = (nodes, pt) => nodes.forEach((n) => setAttrs(n, { cx: pt.x, cy: pt.y, visibility: "visible" }));
    const hide = (nodes) => nodes.forEach((n) => n.setAttribute("visibility", "hidden"));
    function next() {
      if (phase === "in") phase = "up";
      else if (phase === "up") { phase = "split"; answered = false; }
      else if (phase === "split") {
        if (!answered) { showAnswer(t); answered = true; }
        if (t < 8) { t += 1; setLoop(t); phase = "up"; } else phase = "hold";
      } else if (phase === "hold") { phase = "fade"; fig.classList.add("swap"); }
      else { setItem((cur + 1) % items.length); t = 1; setLoop(1); fig.classList.remove("swap"); phase = "in"; }
    }
    function draw() {
      const u = el / dur();
      let py = -99;
      if (phase === "in") { put(main, at(pIn, ease(u))); hide(read); }
      else if (phase === "up") { const pt = at(pUp, u); py = pt.y; put(main, pt); hide(read); }
      else if (phase === "split") {
        if (t < 8) put(main, at(pBack, ease(u))); else hide(main);
        const ur = el / DUR.read;
        if (ur < 1) put(read, at(pRead, ur));
        else {
          if (!answered) { showAnswer(t); answered = true; }
          if (ur < 1.6) put(read, at(pOut, (ur - 1) / 0.6)); else hide(read);
        }
      } else { hide(main); hide(read); }
      layers.forEach((b) => b.node.classList.toggle("on", Math.abs(py - b.cy) < 24));
      readout.classList.toggle("flash", phase === "split" && el >= DUR.read && el < DUR.read + 260);
    }
    function tick(ts) {
      raf = 0;
      if (last !== null) {
        el += Math.min(100, ts - last);
        for (let d = dur(); el >= d; d = dur()) { el -= d; next(); }
      }
      last = ts;
      draw();
      if (playing && visible) raf = requestAnimationFrame(tick);
    }
    function run() { if (playing && visible && !raf) { last = null; raf = requestAnimationFrame(tick); } }
    function setToggle() {
      toggle.textContent = "";
      const g = s("svg", { viewBox: "0 0 12 12", "aria-hidden": "true" }, toggle);
      if (playing) { s("rect", { x: 2, y: 1.5, width: 3, height: 9, rx: 1 }, g); s("rect", { x: 7, y: 1.5, width: 3, height: 9, rx: 1 }, g); } else s("path", { d: "M3 1.5 10.5 6 3 10.5Z" }, g);
      toggle.appendChild(document.createTextNode(playing ? "Pause" : "Play"));
    }
    function go(i, start) {             /* show an item from its first loop */
      fig.classList.remove("swap");
      setItem(i); t = 1; setLoop(1); phase = "in"; el = 0;
      if (start) { playing = true; setToggle(); }
      draw(); run();
    }
    function still(i) {                 /* the item with all eight answers, without motion */
      setItem(i);
      LOOPS.forEach(showAnswer);
      t = 8; setLoop(8); phase = "hold"; el = 0; draw();
    }
    /* One height for all the items, so that the page below does not move when the item changes. */
    function fit() {
      const grid = $(".tz-grid", fig), keep = [cur, t, phase, el, answered];
      let tallest = 0;
      grid.style.minHeight = "0";
      items.forEach((it, i) => {
        still(i);
        const vd = $("#tz-verdict"); vd.className = "verdict"; vd.textContent = "✗ Answers " + LETTER[0] + " (0.00): wrong; the correct option is " + LETTER[it.gold] + ".";
        tallest = Math.max(tallest, grid.offsetHeight);
      });
      grid.style.minHeight = tallest + "px";
      [cur, t, phase, el, answered] = keep;
      setItem(cur);
      const shown = phase === "hold" || phase === "fade" ? 8 : phase === "split" && answered ? t : t - 1;
      for (let k = 1; k <= shown; k++) showAnswer(k);
      setLoop(t); draw();
    }
    let width = window.innerWidth, timer = 0;
    window.addEventListener("resize", () => { if (window.innerWidth === width) return; width = window.innerWidth; clearTimeout(timer); timer = setTimeout(fit, 150); });
    toggle.addEventListener("click", () => {
      if (!playing && phase === "hold") return go(cur, true);
      playing = !playing; setToggle(); run();
    });
    if ("IntersectionObserver" in window) new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; run(); }, { threshold: 0.15 }).observe(fig);
    setToggle();
    if (playing) { fit(); go(0, false); } else { phase = "hold"; fit(); still(0); }
  })();

  /* ---------------------------------------------------------------- tiles */
  (function tiles() {
    const acc = S.perloop.all.acc.v, liars = S.depth.liars.sansi;
    const data = [
      ["+" + f1(S.gain_by_group.all[0]) + " points", "over a single-pass model of the same shape, with the same data and recipe"],
      [f1(acc[7]) + "%", "accuracy after eight loops; " + f1(acc[0]) + "% after one"],
      [f1(S.cost.sansi_per_loop[7]) + "×", "the computation of one pass: the gain is paid in computation, not in parameters"],
      [holds(liars[0]) + " → " + holds(liars[3]), "dependent steps followed on liar chains after one loop and after four"],
    ];
    const box = $("#tiles");
    for (const [v, l] of data) { const t = h("div", { class: "tile" }, box); h("div", { class: "v" }, t, v); h("div", { class: "l" }, t, l); }
  })();

  /* ---------------------------------------------------------------- accuracy against parameters */
  (function sizeChart() {
    const host = $("#ch-size"), W = 460, H = 380, Z = S.size;
    legend(host, [{ kind: "line", marker: "dot", color: COL.sansi, label: "SanSi (looped)" }, { kind: "line", marker: "square", dash: "5 4", color: COL.qwen, label: "Qwen3.5" },
      { kind: "mark", marker: "square", color: COL.smol, label: "SmolLM2" }, { kind: "mark", marker: "triangle", color: COL.kev, label: "Kev-4B (our data)" }, { kind: "line", color: COL.jev, dash: "6 4", width: 1.5, label: "Jev API" }]);
    const c = base(host, W, H, { t: 30, r: 16, b: 48, l: 44 }, "Accuracy against the number of parameters of the backbone");
    const x = logScale(1.2, 5.0, c.x0, c.x1), y = lin(54, 82, c.y1, c.y0), rows = [];
    yAxis(c, y, [56, 60, 64, 68, 72, 76, 80], String, "Accuracy (%)");
    xAxis(c, x, [1.5, 2, 3, 4], (v) => v + "B", "Parameters of the backbone");
    s("line", { x1: c.x0, x2: c.x1, y1: y(Z.jev), y2: y(Z.jev), stroke: COL.jev, "stroke-width": 1.5, "stroke-dasharray": "6 4" }, c.svg);
    s("text", { class: "lab", x: c.x0 + 6, y: y(Z.jev) - 6 }, c.svg, "Jev API, " + f1(Z.jev) + "% (size not public)");
    const sp = Object.fromEntries(Z.single_pass.map((d) => [d.arm, d]));
    s("line", { x1: x(sp.qwen2b.params), y1: y(sp.qwen2b.acc[0]), x2: x(sp.qwen.params), y2: y(sp.qwen.acc[0]), stroke: COL.qwen, "stroke-width": 2, "stroke-dasharray": "5 4" }, c.svg);
    s("line", { x1: x(Z.sansi.params), y1: y(Z.sansi.acc.v[7]), x2: x(Z.sansi26.params), y2: y(Z.sansi26.acc.v[7]), stroke: COL.sansi, "stroke-width": 2 }, c.svg);
    for (const [m, name] of [[Z.sansi, "SanSi"], [Z.sansi26, "SanSi-2.6B"]]) {
      const px = x(m.params);
      s("line", { x1: px, y1: y(m.acc.v[0]), x2: px, y2: y(m.acc.v[7]), stroke: COL.sansi, "stroke-width": 2 }, c.svg);
      for (const t of [1, 2, 8]) {
        hover(dot(c.svg, px, y(m.acc.v[t - 1]), t === 8 ? COL.dark : COL.sansi, { r: t === 8 ? 5.5 : 4.5 }), name + ", read after loop " + t, [{ label: "Accuracy", value: f1(m.acc.v[t - 1]) + " ± " + f1(m.acc.sd[t - 1]) + "%" }, { label: "Parameters", value: m.params + "B" }]);
        rows.push([name, m.params + "B", String(t), f1(m.acc.v[t - 1]) + " ± " + f1(m.acc.sd[t - 1])]);
      }
    }
    for (const arm of ["smol", "qwen2b", "qwen", "kevrep"]) {
      const d = sp[arm], px = x(d.params), col = arm === "smol" ? COL.smol : arm === "kevrep" ? COL.kev : COL.qwen;
      const mark = arm === "kevrep" ? triangle(c.svg, px, y(d.acc[0]) - 1, col) : square(c.svg, px, y(d.acc[0]), col, { size: 10 });
      hover(mark, d.name + ", one pass", [{ label: "Accuracy", value: f1(d.acc[0]) + " ± " + f1(d.acc[1]) + "%" }, { label: "Parameters", value: d.params + "B" }]);
      if (arm === "kevrep") s("text", { class: "lab", x: px + 16, y: y(d.acc[0]) - 14, "text-anchor": "end" }, c.svg, d.name);
      else s("text", { class: "lab", x: px, y: y(d.acc[0]) + 20, "text-anchor": "middle" }, c.svg, d.name);
      rows.push([d.name, d.params + "B", "1", f1(d.acc[0]) + " ± " + f1(d.acc[1])]);
    }
    const sx = x(Z.sansi.params), a = Z.sansi.acc.v;
    s("text", { class: "strong", x: sx - 10, y: y(a[7]) - 11, "text-anchor": "end" }, c.svg, "SanSi");
    s("text", { class: "lab", x: sx - 10, y: y(a[7]) + 5, "text-anchor": "end" }, c.svg, "loop 8");
    s("text", { class: "lab", x: sx - 10, y: y(a[1]) + 4, "text-anchor": "end" }, c.svg, "loop 2");
    s("text", { class: "lab", x: sx - 10, y: y(a[0]) + 4, "text-anchor": "end" }, c.svg, "loop 1");
    s("text", { class: "strong", x: x(Z.sansi26.params), y: y(Z.sansi26.acc.v[7]) - 12, "text-anchor": "middle" }, c.svg, "SanSi-2.6B");
    rows.push(["Jev API (not trained on our data)", "not public", "1", f1(Z.jev)]);
    table(host, ["Model", "Parameters", "Loops", "Accuracy (%)"], rows);
  })();

  /* ---------------------------------------------------------------- accuracy against computation */
  (function costChart() {
    const host = $("#ch-cost"), W = 460, H = 360;
    legend(host, [{ kind: "line", marker: "dot", color: COL.sansi, label: "SanSi, after loops 1–8" }, { kind: "line", marker: "square", dash: "5 4", color: COL.qwen, label: "Qwen3.5" }, { kind: "mark", marker: "square", color: COL.smol, label: "SmolLM2" },
      { kind: "mark", marker: "triangle", color: COL.kev, label: "Kev-4B (our data)" }]);
    const c = base(host, W, H, { t: 30, r: 16, b: 48, l: 44 }, "Accuracy against the computation spent on a decision");
    const x = logScale(0.62, 9.2, c.x0, c.x1), y = lin(54, 78, c.y1, c.y0), acc = S.perloop.all.acc, cost = S.cost.sansi_per_loop;
    yAxis(c, y, [54, 60, 66, 72, 78], String, "Accuracy (%)");
    xAxis(c, x, [1, 2, 3, 4, 6, 8], (v) => v + "×", "GPU time per decision (one loop = 1)");
    const pts = LOOPS.map((t) => [x(cost[t - 1]), y(acc.v[t - 1])]);
    s("polygon", { points: LOOPS.map((t) => `${x(cost[t - 1])},${y(acc.v[t - 1] + acc.sd[t - 1])}`).concat(LOOPS.map((t) => `${x(cost[t - 1])},${y(acc.v[t - 1] - acc.sd[t - 1])}`).reverse()).join(" "), fill: COL.sansi, opacity: 0.12 }, c.svg);
    s("path", { d: pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" "), fill: "none", stroke: COL.sansi, "stroke-width": 2, "stroke-linejoin": "round" }, c.svg);
    const qa = ["qwen2b", "qwen"].map((a) => [x(S.cost.single_pass[a]), y(ref(a, "all", "acc"))]);
    s("line", { x1: qa[0][0], y1: qa[0][1], x2: qa[1][0], y2: qa[1][1], stroke: COL.qwen, "stroke-width": 2, "stroke-dasharray": "5 4" }, c.svg);
    const rows = [];
    for (const arm of ["smol", "qwen2b", "qwen", "kevrep"]) {
      const v = S.refs[arm].all.acc, px = x(S.cost.single_pass[arm]), col = arm === "smol" ? COL.smol : arm === "kevrep" ? COL.kev : COL.qwen;
      const mark = arm === "kevrep" ? triangle(c.svg, px, y(v[0]) - 1, col) : square(c.svg, px, y(v[0]), col, { size: 10 });
      hover(mark, S.names[arm] + ", one pass", [{ label: "Accuracy", value: f1(v[0]) + " ± " + f1(v[1]) + "%" }, { label: "GPU time", value: f2(S.cost.single_pass[arm]) + "×" }]);
      const at = { smol: [9, 19, "start"], qwen2b: [0, 19, "middle"], qwen: [11, -3, "start"], kevrep: [-9, -8, "end"] }[arm];
      s("text", { class: "lab", x: px + at[0], y: y(v[0]) + at[1], "text-anchor": at[2] }, c.svg, S.names[arm]);
      rows.push([S.names[arm], "1", f2(S.cost.single_pass[arm]), f1(v[0])]);
    }
    LOOPS.forEach((t, i) => {
      hover(dot(c.svg, pts[i][0], pts[i][1], t === 8 ? COL.dark : COL.sansi, { r: t === 8 ? 5.5 : 4.5 }), "SanSi, read after loop " + t, [{ label: "Accuracy", value: f1(acc.v[i]) + " ± " + f1(acc.sd[i]) + "%" }, { label: "GPU time", value: f2(cost[i]) + "×" }]);
      rows.push(["SanSi", String(t), f2(cost[i]), f1(acc.v[i])]);
    });
    s("text", { class: "lab", x: pts[0][0] - 9, y: pts[0][1] + 4, "text-anchor": "end" }, c.svg, "loop 1");
    s("text", { class: "lab", x: pts[2][0] + 8, y: pts[2][1] + 16 }, c.svg, "loop 3");
    s("text", { class: "strong", x: pts[7][0], y: pts[7][1] - 12, "text-anchor": "middle" }, c.svg, "loop 8");
    table(host, ["Model", "Loops", "GPU time (×)", "Accuracy (%)"], rows);
  })();

  /* ---------------------------------------------------------------- the main comparison as a table */
  (function mainTable() {
    const t = h("table", { class: "main" }, $("#tbl-main")), hr = h("tr", {}, h("thead", {}, t)), tb = h("tbody", {}, t), pm = (v, f) => f(v[0]) + " ± " + f(v[1]);
    ["Model", "Params", "Loops", "Cost", "All", "In-dist.", "Near", "Far", "ECE ↓", "Evid. AUROC ↑"].forEach((x) => h("th", { scope: "col" }, hr, x));
    for (const r of S.main) {
      const tr = h("tr", r.ours ? { class: "ours" } : {}, tb);
      [r.name, r.params, String(r.loops), r.cost === null ? "–" : f1(r.cost) + "×", pm(r.acc.all, f1), pm(r.acc.in_dist, f1), pm(r.acc.near, f1), pm(r.acc.far, f1), pm(r.ece, f3), pm(r.evid, f3)].forEach((v, i) => h(i ? "td" : "th", i ? (i === 4 ? { class: "all" } : {}) : { scope: "row" }, tr, v));
    }
  })();

  /* ---------------------------------------------------------------- Kev on our data; released models on items neither side trained on */
  (function kev() {
    const K = S.kev, D = K.diffs, acc = (arm) => f1(S.refs[arm].all.acc[0]);
    $("#kev-text").textContent = "Kev is the open reimplementation of Jev's design. To compare at equal data, we trained Kev's own code on our " + (12800).toLocaleString("en-US") + " training items. This model, Kev-4B (our data), reaches " + acc("kevrep") +
      "%. It cannot be separated from the same backbone trained with our recipe (Qwen3.5-4B, " + acc("qwen") + "%; difference " + f1(D["kevrep-qwen"][0]) + " points " + iv(D["kevrep-qwen"]) + "). SanSi is " + f1(D["kevrep-sansi"][0]) + " points below it " + iv(D["kevrep-sansi"]) +
      " with a third of its parameters, and SanSi-2.6B is " + f1(D["sansi26-kevrep"][0]) + " points above it " + iv(D["sansi26-kevrep"]) + " with 63% of its parameters.";
    $("#kev-text2").textContent = "The released Kev-4B and the Jev API were not trained on our data, and " + K.counts.kev_train + " of our test items are training items of the released Kev. A comparison that favours neither side is therefore only possible on the " +
      K.counts.neither.toLocaleString("en-US") + " test items of far transfer and JevBench, whose sources no model was trained on.";
    const host = $("#ch-neither"), groups = [["Released models, not trained on our data", K.neither.filter((r) => r.released)], ["Trained on our data", K.neither.filter((r) => !r.released)]];
    const W = 900, rowH = 30, headH = 28, n = K.neither.length, H = 16 + groups.length * headH + n * rowH + 44, c = base(host, W, H, { t: 16, r: 90, b: 44, l: 190 }, "Accuracy on the test items that neither side trained on");
    const x = lin(60, 90, c.x0, c.x1), rows = [];
    for (const t of [60, 70, 80, 90]) { s("line", { class: "grid", x1: x(t), x2: x(t), y1: c.y0, y2: c.y1 }, c.svg); s("text", { x: x(t), y: c.y1 + 17, "text-anchor": "middle" }, c.svg, String(t)); }
    s("text", { class: "lab", x: (c.x0 + c.x1) / 2, y: c.y1 + 36, "text-anchor": "middle" }, c.svg, "Accuracy (%) on " + K.counts.neither.toLocaleString("en-US") + " items");
    let cy = c.y0;
    const marks = { jev: (px, py) => s("path", { d: `M${px} ${py - 6.5} L${px + 6.5} ${py} L${px} ${py + 6.5} L${px - 6.5} ${py} Z`, fill: COL.jev, stroke: COL.surface, "stroke-width": 2 }, c.svg),
      kev4b: (px, py) => triangle(c.svg, px, py, COL.kev, { hollow: true }), kevrep: (px, py) => triangle(c.svg, px, py, COL.kev), qwen: (px, py) => square(c.svg, px, py, COL.qwen, { size: 10 }),
      sansi26: (px, py) => dot(c.svg, px, py, COL.dark, { r: 5.5 }), all8: (px, py) => dot(c.svg, px, py, COL.sansi, { r: 5.5 }) };
    for (const [title, list] of groups) {
      s("text", { x: 2, y: cy + 18 }, c.svg, title); cy += headH;
      for (const r of list) {
        const py = cy + rowH / 2, px = x(r.acc);
        s("text", { class: "lab", x: c.x0 - 12, y: py + 4, "text-anchor": "end" }, c.svg, r.name);
        s("line", { x1: c.x0, x2: px, y1: py, y2: py, stroke: COL.grayLight, "stroke-width": 1 }, c.svg);
        if (r.sd) s("line", { x1: x(r.acc - r.sd), x2: x(r.acc + r.sd), y1: py, y2: py, stroke: COL.ink2, "stroke-width": 1.5 }, c.svg);
        marks[r.arm](px, py);
        s("text", { class: "strong", x: px + 14, y: py + 4 }, c.svg, f1(r.acc));
        hover(s("rect", { x: 0, y: cy, width: W, height: rowH, fill: "transparent" }, c.svg), r.name, [{ label: "Accuracy on these items", value: f1(r.acc) + (r.sd ? " ± " + f1(r.sd) : "") + "%" }, { label: r.released ? "Not trained on our data; one run" : "Trained on our data; three seeds", value: "" }], true);
        rows.push([r.name, r.released ? "no" : "yes", f1(r.acc) + (r.sd ? " ± " + f1(r.sd) : "")]);
        cy += rowH;
      }
    }
    table(host, ["Model", "Trained on our data", "Accuracy (%)"], rows);
  })();

  /* ---------------------------------------------------------------- by item type */
  (function typesChart() {
    const host = $("#ch-types"), types = S.types.slice().sort((a, b) => b.gain[0] - a.gain[0]);
    legend(host, [{ kind: "mark", marker: "square", color: COL.smol, label: "SmolLM2-1.7B (one pass)" }, { kind: "mark", marker: "dot", color: COL.sansi, hollow: true, label: "SanSi, loop 1" },
      { kind: "mark", marker: "dot", color: COL.dark, label: "SanSi, loop 8" }, { kind: "mark", marker: "tick", color: COL.qwen, label: "Qwen3.5-4B (one pass)" }]);
    const W = 900, rowH = 34, H = 34 + rowH * types.length + 44, c = base(host, W, H, { t: 34, r: 78, b: 44, l: 170 }, "Accuracy by item type");
    const x = lin(44, 86, c.x0, c.x1);
    for (const t of [50, 60, 70, 80]) { s("line", { class: "grid", x1: x(t), x2: x(t), y1: c.y0 - 6, y2: c.y1 }, c.svg); s("text", { x: x(t), y: c.y1 + 17, "text-anchor": "middle" }, c.svg, String(t)); }
    s("text", { class: "lab", x: (c.x0 + c.x1) / 2, y: c.y1 + 36, "text-anchor": "middle" }, c.svg, "Accuracy (%)");
    s("text", { class: "lab", x: c.x1 + 14, y: c.y0 - 12 }, c.svg, "Gain");
    const rows = [];
    types.forEach((t, i) => {
      const cy = c.y0 + rowH * (i + 0.5), l1 = t.sansi.v[0], l8 = t.sansi.v[7];
      s("text", { class: "lab", x: c.x0 - 12, y: cy + 4, "text-anchor": "end" }, c.svg, t.name);
      s("line", { x1: x(t.smol[0]), x2: x(l8), y1: cy, y2: cy, stroke: COL.sansi, "stroke-width": 2, opacity: 0.45 }, c.svg);
      s("line", { x1: x(t.qwen[0]), x2: x(t.qwen[0]), y1: cy - 9, y2: cy + 9, stroke: COL.qwen, "stroke-width": 3 }, c.svg);
      square(c.svg, x(t.smol[0]), cy, COL.smol, { size: 10 });
      dot(c.svg, x(l1), cy, COL.sansi, { hollow: true });
      dot(c.svg, x(l8), cy, COL.dark, { r: 5.5 });
      s("text", { class: "strong", x: c.x1 + 14, y: cy + 4 }, c.svg, signed(t.gain[0]));
      const hitRow = s("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, c.svg);
      hover(hitRow, t.name + " (" + t.n.toLocaleString("en-US") + " items)", [
        { color: COL.smol, label: "SmolLM2-1.7B", value: f1(t.smol[0]) + "%" }, { color: COL.sansi, label: "SanSi, loop 1", value: f1(l1) + "%" },
        { color: COL.dark, label: "SanSi, loop 8", value: f1(l8) + "%" }, { color: COL.qwen, label: "Qwen3.5-4B", value: f1(t.qwen[0]) + "%" },
        { label: "Gain over SmolLM2 [95% interval]", value: signed(t.gain[0]) + " [" + f1(t.gain[1]) + ", " + f1(t.gain[2]) + "]" }], true);
      rows.push([t.name, t.n.toLocaleString("en-US"), f1(t.smol[0]), f1(l1), f1(l8), f1(t.qwen[0]), signed(t.gain[0])]);
    });
    table(host, ["Item type", "Items", "SmolLM2-1.7B", "SanSi, loop 1", "SanSi, loop 8", "Qwen3.5-4B", "Gain (points)"], rows);
    const g = S.gain_by_group;
    h("p", { class: "note", style: "margin:10px 0 0" }, host, "By distance from the training data, the gain over SmolLM2-1.7B is " + f1(g.in_dist[0]) + " points on in-distribution items, " + f1(g.near[0]) + " on near transfer and " + f1(g.far[0]) + " on far transfer.");
  })();

  /* ---------------------------------------------------------------- loop by loop */
  (function perLoop() {
    const METRIC = {
      acc: { name: "Accuracy", short: "Accuracy", title: "Accuracy (%)", fmt: f1, sub: "The option with the largest probability is the correct one.", refs: true },
      conf: { name: "Mean confidence of right and wrong answers", short: "Confidence", title: "Mean confidence", fmt: f3, sub: "Confidence is the probability of the chosen option. It rises on right and on wrong answers alike.", refs: false },
      ece: { name: "Calibration error (ECE)", short: "Calibration", title: "ECE (lower is better)", fmt: f3, sub: "Expected calibration error on the answerable items: the gap between confidence and accuracy.", refs: true },
      auroc_evid: { name: "Noticing missing evidence (evidence AUROC)", short: "Missing evidence", title: "Evidence AUROC", fmt: f3, sub: "How well confidence separates items with their key evidence from the same items without it.", refs: true },
      hard: { name: "Hard answers when the evidence is missing", short: "Hard answers", title: "Hard answers (%, lower is better)", fmt: f1, sub: "Share of the items without their key evidence on which the model still commits to an answer.", refs: true },
    };
    const GROUP = { all: "All test items", in_dist: "In distribution", near: "Near transfer", far: "Far transfer" };
    const mSel = $("#pl-metric"), gSel = $("#pl-group");
    for (const [k, v] of Object.entries(METRIC)) h("option", { value: k }, mSel, v.name);
    for (const [k, v] of Object.entries(GROUP)) h("option", { value: k }, gSel, v);
    segment(mSel, "seg", Object.fromEntries(Object.entries(METRIC).map(([k, v]) => [k, v.short])));
    segment(gSel, "seg");
    function render() {
      const mk = mSel.value, gk = gSel.value, M = METRIC[mk], P = S.perloop[gk], host = $("#ch-perloop");
      $("#pl-title").textContent = M.name + " after every loop: " + GROUP[gk].toLowerCase();
      const cd = Object.values(S.calib_dev).map((v) => v[1]);
      $("#pl-sub").textContent = M.sub + " Line: mean of three seeds; band: one standard deviation." + (M.refs ? " Horizontal lines: single-pass models, run once." : "") +
        (mk === "ece" ? " These are the probabilities as trained. With one temperature fitted on the development set, the ECE of SanSi, SanSi-2.6B, Qwen3.5-4B and Kev-4B (our data) all fall to " + f3(Math.min(...cd)) + "–" + f3(Math.max(...cd)) + "." : "");
      let series, refs = [], all = [];
      if (mk === "conf") {
        series = [{ name: "Right answers", short: "right answers", color: COL.sansi, values: P.conf_right.v, sd: P.conf_right.sd }, { name: "Wrong answers", short: "wrong answers", color: COL.sansi, dash: "5 4", hollow: true, values: P.conf_wrong.v, sd: P.conf_wrong.sd }];
      } else {
        series = [{ name: "SanSi", color: COL.sansi, values: P[mk].v, sd: P[mk].sd }];
        refs = [{ arm: "kevrep", color: COL.kev }, { arm: "qwen", color: COL.qwen }, { arm: "qwen2b", color: COL.qwen, dash: "5 4" }, { arm: "smol", color: COL.smol }].map((r) => ({ name: S.names[r.arm], color: r.color, dash: r.dash, value: ref(r.arm, gk, mk) }));
      }
      series.forEach((se) => se.values.forEach((v, i) => all.push(v - se.sd[i], v + se.sd[i])));
      refs.forEach((r) => all.push(r.value));
      const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08, nt = niceTicks(lo - pad, hi + pad, 5);
      lineChart(host, { W: 900, H: 340, xs: LOOPS, xDomain: [0.6, 8.4], xTitle: "Loop", yDomain: nt.domain, yTicks: nt.ticks, yFmt: M.fmt, yTitle: M.title, series, refs, endLabels: true, tipTitle: (t) => "Loop " + t, aria: M.name + " of SanSi after every loop" });
      const head = ["Loop"].concat(series.map((se) => se.name)), rows = LOOPS.map((t, i) => [String(t)].concat(series.map((se) => M.fmt(se.values[i]) + " ± " + M.fmt(se.sd[i]))));
      refs.forEach((r) => rows.push([r.name + " (one pass)", M.fmt(r.value)]));
      table(host, head, rows);
    }
    mSel.addEventListener("change", render); gSel.addEventListener("change", render);
    render();
  })();

  /* ---------------------------------------------------------------- answers fixed and broken */
  (function fixedChart() {
    const host = $("#ch-fixed"), D = S.dynamics.from_first, W = 460, H = 320;
    legend(host, [{ kind: "rect", color: COL.sansi, label: "fixed (wrong → right)" }, { kind: "rect", color: COL.gray, label: "broken (right → wrong)" }]);
    const c = base(host, W, H, { t: 30, r: 14, b: 48, l: 44 }, "Answers fixed and broken since loop 1");
    const Ts = [2, 3, 4, 5, 6, 7, 8], x = lin(1.4, 8.6, c.x0, c.x1), y = lin(-10, 25, c.y1, c.y0);
    yAxis(c, y, [-10, 0, 10, 20], (v) => String(Math.abs(v)), "Share of items (%)");
    for (const t of Ts) s("text", { x: x(t), y: c.y1 + 17, "text-anchor": "middle" }, c.svg, String(t));
    s("text", { class: "lab", x: (c.x0 + c.x1) / 2, y: c.y1 + 36, "text-anchor": "middle" }, c.svg, "Loop T");
    const rows = [];
    for (const t of Ts) {
      const fx = D.fixed[t - 1], br = D.broken[t - 1];
      column(c.svg, x(t), 22, y(0) - 1, y(fx), COL.sansi);
      column(c.svg, x(t), 22, y(0) + 1, y(-br), COL.gray);
      hover(s("rect", { x: x(t) - 24, y: c.y0, width: 48, height: c.y1 - c.y0, fill: "transparent" }, c.svg), "Loop " + t + " against loop 1",
        [{ color: COL.sansi, label: "Fixed", value: f1(fx) + "%" }, { color: COL.gray, label: "Broken", value: f1(br) + "%" }, { label: "Net gain", value: signed(fx - br) + " points" }], true);
      rows.push([String(t), f1(fx), f1(br), signed(fx - br)]);
    }
    s("line", { class: "axis", x1: c.x0, x2: c.x1, y1: y(0), y2: y(0) }, c.svg);
    s("text", { class: "strong", x: x(8), y: y(D.fixed[7]) - 7, "text-anchor": "middle" }, c.svg, f1(D.fixed[7]));
    s("text", { class: "strong", x: x(8), y: y(-D.broken[7]) + 15, "text-anchor": "middle" }, c.svg, f1(D.broken[7]));
    table(host, ["Loop T", "Fixed (%)", "Broken (%)", "Net (points)"], rows);
  })();

  (function settleChart() {
    const host = $("#ch-settle"), D = S.dynamics, W = 460, H = 320;
    legend(host, [{ kind: "rect", color: COL.sansi, label: "answers that settle at this loop" }, { kind: "line", marker: "dot", color: COL.dark, label: "of these, right at loop 8" }]);
    const c = base(host, W, H, { t: 30, r: 14, b: 48, l: 44 }, "When answers settle and how often those answers are right");
    const x = lin(0.4, 8.6, c.x0, c.x1), y = lin(0, 100, c.y1, c.y0);
    yAxis(c, y, [0, 25, 50, 75, 100], String, "Share (%)");
    xAxis(c, x, LOOPS, String, "Loop at which the answer settles");
    const pts = [], rows = [];
    LOOPS.forEach((t, i) => {
      column(c.svg, x(t), 22, c.y1, y(D.settle_dist[i]), COL.sansi);
      pts.push([x(t), y(D.by_settle.acc8[i])]);
      rows.push([String(t), f1(D.settle_dist[i]), f1(D.by_settle.acc8[i]), f2(D.by_settle.conf8[i])]);
    });
    s("path", { d: pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" "), fill: "none", stroke: COL.dark, "stroke-width": 2, "stroke-linejoin": "round" }, c.svg);
    LOOPS.forEach((t, i) => {
      dot(c.svg, pts[i][0], pts[i][1], COL.dark);
      hover(s("rect", { x: x(t) - 22, y: c.y0, width: 44, height: c.y1 - c.y0, fill: "transparent" }, c.svg), t === 1 ? "Answers that never change" : "Answers that settle at loop " + t,
        [{ color: COL.sansi, label: "Share of the items", value: f1(D.settle_dist[i]) + "%" }, { color: COL.dark, label: "Right at loop 8", value: f1(D.by_settle.acc8[i]) + "%" }, { label: "Mean confidence at loop 8", value: f2(D.by_settle.conf8[i]) }], true);
    });
    s("text", { class: "strong", x: x(1) + 16, y: y(D.settle_dist[0]) + 4 }, c.svg, f1(D.settle_dist[0]) + "%");
    s("text", { class: "strong", x: x(8), y: pts[7][1] - 10, "text-anchor": "middle" }, c.svg, f1(D.by_settle.acc8[7]) + "%");
    table(host, ["Settles at loop", "Share of items (%)", "Right at loop 8 (%)", "Mean confidence"], rows);
  })();

  /* ---------------------------------------------------------------- more loops than trained */
  (function nloopsChart() {
    const host = $("#ch-nloops"), xs = [], a8 = S.nloops.all8, a4 = S.nloops.all4;
    for (let t = 2; t <= 16; t++) xs.push(t);
    const pick = (m, n) => xs.map((t) => (t <= n ? m[t - 1] : null));
    const series = [{ name: "Trained with 8 loops (SanSi)", short: "8 trained loops", color: COL.sansi, values: pick(a8.v, 16), sd: pick(a8.sd, 16).map((v) => v || 0), hollowFrom: 8 },
      { name: "Trained with 4 loops", short: "4 trained loops", color: COL.gray, values: pick(a4.v, 8), sd: pick(a4.sd, 8).map((v) => v || 0), hollowFrom: 4 }];
    lineChart(host, { W: 900, H: 320, xs, xDomain: [1.5, 16.5], xTitle: "Loop", yDomain: [66, 74], yTicks: [66, 68, 70, 72, 74], yFmt: f1, yTitle: "Accuracy (%)", series, endLabels: true,
      regions: [{ from: 8.5, to: 16.5, label: "beyond the eight trained loops" }], tipTitle: (t) => "Loop " + t, aria: "Accuracy when a model is run for more loops than it was trained with" });
    table(host, ["Loop", "Trained with 8 loops", "Trained with 4 loops"], [1].concat(xs).map((t) => [String(t), f1(a8.v[t - 1]) + " ± " + f1(a8.sd[t - 1]), t <= 8 ? f1(a4.v[t - 1]) + " ± " + f1(a4.sd[t - 1]) : null]));
  })();

  /* ---------------------------------------------------------------- reasoning depth */
  (function depth() {
    const tSel = $("#dp-task"), lSel = $("#dp-loop"), KS = S.depth.liars.k;
    for (const [k, v] of Object.entries(S.depth)) h("option", { value: k }, tSel, v.name);
    segment(tSel, "seg");
    const ramp = (t) => {             /* one hue, light to dark: accuracy above chance */
      const stops = [[241, 245, 250], [74, 152, 227], [5, 41, 92]], u = Math.max(0, Math.min(1, t)) * 2, i = Math.min(1, Math.floor(u)), f = u - i;
      return "rgb(" + stops[i].map((a, j) => Math.round(a + (stops[i + 1][j] - a) * f)).join(",") + ")";
    };
    function render() {
      const D = S.depth[tSel.value], loop = +lSel.value, acc = D.sansi[loop - 1];
      $("#dp-now").textContent = String(loop);
      $("#dp-title").textContent = D.name + ": accuracy by depth";
      $("#dp-sub").textContent = "SanSi after loop " + loop + " holds depth " + holds(acc) + " (at least 75% at every depth up to it); Qwen3.5-4B holds depth " + holds(D.qwen) + ". Means of three seeds.";
      const lo = D.chance - 10;
      lineChart($("#ch-depth"), { W: 460, H: 340, xs: KS, xDomain: [0.4, 16.6], xTicks: [1, 4, 8, 12, 16], xTitle: "Depth k (dependent steps)", yDomain: [lo, 100], yTicks: [D.chance].concat([40, 60, 80, 100].filter((v) => v > D.chance + 5)), yFmt: f1, yTitle: "Accuracy (%)",
        series: [{ name: "SanSi, loop " + loop, color: COL.sansi, values: acc }, { name: "Qwen3.5-4B (one pass)", color: COL.qwen, values: D.qwen, hollow: true }],
        refs: [{ name: "Chance", color: COL.gray, value: D.chance }], regions: [{ from: 8.5, to: 16.6, label: "not seen in training" }], tipTitle: (k) => "Depth " + k, aria: D.name + ": accuracy by depth" });
      table($("#ch-depth"), ["Depth k"].concat(KS.map(String)), [["SanSi, loop " + loop].concat(acc.map(f1)), ["Qwen3.5-4B"].concat(D.qwen.map(f1))]);

      const host = $("#ch-heat"); host.textContent = "";
      const W = 460, H = 356, c = base(host, W, H, { t: 16, r: 12, b: 70, l: 44 }, D.name + ": accuracy of SanSi at every loop and depth");
      const cw = (c.x1 - c.x0) / 16, ch = (c.y1 - c.y0) / 16;
      for (let t = 1; t <= 16; t++) {
        const cy = c.y1 - t * ch;
        if (t % 2 === 0 || t === 1) s("text", { x: c.x0 - 8, y: cy + ch / 2 + 4, "text-anchor": "end" }, c.svg, String(t));
        KS.forEach((k, j) => {
          const v = D.sansi[t - 1][j], cell = s("rect", { x: c.x0 + j * cw + 1, y: cy + 1, width: cw - 2, height: ch - 2, rx: 1.5, fill: ramp((v - D.chance) / (100 - D.chance)) }, c.svg);
          cell.addEventListener("pointermove", (e) => { tipFill("Loop " + t + ", depth " + k, [{ label: "Accuracy", value: f1(v) + "%" }]); tipAt(e.clientX, e.clientY); });
          cell.addEventListener("pointerleave", tipHide);
          cell.addEventListener("click", () => { lSel.value = String(t); render(); });
          cell.style.cursor = "pointer";
        });
      }
      for (const k of [1, 4, 8, 12, 16]) s("text", { x: c.x0 + (k - 0.5) * cw, y: c.y1 + 17, "text-anchor": "middle" }, c.svg, String(k));
      s("text", { class: "lab", x: (c.x0 + c.x1) / 2, y: c.y1 + 34, "text-anchor": "middle" }, c.svg, "Depth k");
      s("text", { class: "lab", x: 2, y: c.y0 - 4 }, c.svg, "Loop");
      s("rect", { x: c.x0 - 1, y: c.y1 - loop * ch - 1, width: c.x1 - c.x0 + 2, height: ch + 2, fill: "none", stroke: COL.ink2, "stroke-width": 1.5, rx: 2 }, c.svg);
      s("line", { x1: c.x0 + 8 * cw, x2: c.x0 + 8 * cw, y1: c.y0, y2: c.y1, stroke: COL.surface, "stroke-width": 3 }, c.svg);
      s("line", { x1: c.x0, x2: c.x1, y1: c.y1 - 8 * ch, y2: c.y1 - 8 * ch, stroke: COL.surface, "stroke-width": 3 }, c.svg);
      const gx = c.x0, gy = c.y1 + 46, gw = 150, id = "ramp-" + tSel.value, grad = s("linearGradient", { id }, s("defs", {}, c.svg));
      [0, 0.5, 1].forEach((u) => s("stop", { offset: u * 100 + "%", "stop-color": ramp(u) }, grad));
      s("rect", { x: gx, y: gy, width: gw, height: 9, rx: 2, fill: "url(#" + id + ")" }, c.svg);
      s("text", { x: gx, y: gy + 22 }, c.svg, "chance (" + D.chance + "%)");
      s("text", { x: gx + gw, y: gy + 22, "text-anchor": "end" }, c.svg, "100%");
      table(host, ["Loop"].concat(KS.map((k) => "k=" + k)), D.sansi.map((row, i) => [String(i + 1)].concat(row.map(f1))));
    }
    tSel.addEventListener("change", render); lSel.addEventListener("input", render);
    render();
  })();

  /* ---------------------------------------------------------------- the verifier use case */
  (function verifier() {
    const host = $("#ch-verifier"), V = S.verifier, Ts = [1, 2, 4, 8], W = 900, H = 320;
    const c = base(host, W, H, { t: 30, r: 150, b: 48, l: 46 }, "F1 of the generator by the loop at which SanSi is read");
    const x = lin(-0.6, 3.6, c.x0, c.x1), y = lin(25, 52, c.y1, c.y0);
    yAxis(c, y, [25, 30, 35, 40, 45, 50], String, "F1 on the test questions");
    s("line", { class: "axis", x1: c.x0, x2: c.x1, y1: c.y1, y2: c.y1 }, c.svg);
    s("text", { class: "lab", x: (c.x0 + c.x1) / 2, y: c.y1 + 36, "text-anchor": "middle" }, c.svg, "Loop at which the verifier is read");
    s("line", { x1: c.x0, x2: c.x1, y1: y(V.base), y2: y(V.base), stroke: COL.gray, "stroke-width": 1.5 }, c.svg);
    s("text", { class: "lab", x: c.x1 + 10, y: y(V.base) + 4 }, c.svg, "before training: " + f1(V.base));
    const rows = [["Before training", f1(V.base), "", ""]];
    Ts.forEach((t, i) => {
      const d = V.loops[String(t)], px = x(i);
      s("text", { x: px, y: c.y1 + 17, "text-anchor": "middle" }, c.svg, String(t));
      s("line", { x1: px, x2: px, y1: y(V.base), y2: y(d.f1), stroke: COL.sansi, "stroke-width": 2, opacity: 0.45 }, c.svg);
      d.seeds.forEach((v, j) => dot(c.svg, px + (j - 1) * 13, y(v), COL.sansi, { r: 3.5, opacity: 0.55 }));
      dot(c.svg, px, y(d.f1), COL.dark, { r: 6 });
      s("text", { class: "strong", x: px + 26, y: y(d.f1) + 4 }, c.svg, f1(d.f1));
      hover(s("rect", { x: px - 50, y: c.y0, width: 100, height: c.y1 - c.y0, fill: "transparent" }, c.svg), "Reward read after loop " + t,
        [{ color: COL.dark, label: "F1, mean of three runs", value: f1(d.f1) }, { color: COL.sansi, label: "The runs", value: d.seeds.map(f1).join(", ") }, { label: "Change against no training", value: signed(d.f1 - V.base) }], true);
      rows.push(["SanSi read after loop " + t, f1(d.f1), d.seeds.map(f1).join(", "), signed(d.f1 - V.base)]);
    });
    table(host, ["Reward", "F1 (mean)", "F1 of the three runs", "Change"], rows);
  })();

  /* ---------------------------------------------------------------- the item explorer */
  (function explorer() {
    const TIER = { in_dist: "in distribution", near: "near transfer", far: "far transfer" };
    const GROUPS = [["paper", "Examples from the paper"], ["fixed", "Fixed by the loops"], ["broken", "Broken by the loops"], ["always_right", "Right at every loop"], ["always_wrong", "Wrong at every loop"], ["back_and_forth", "Back and forth"], ["unans", "Key evidence removed"]];
    const LABEL = { fixed: "fixed by the loops", broken: "broken by the loops", always_right: "right at every loop", always_wrong: "wrong at every loop", back_and_forth: "back and forth", unans: "key evidence removed" };
    const ONE = [["SmolLM2-1.7B", COL.smol], ["Qwen3.5-2B", COL.qwen], ["Qwen3.5-4B", COL.qwen]];
    const LETTER = "ABCDEFGH";
    const grp = (it) => (it.unans ? "unans" : it.group);
    const inGroup = (it, g) => (g === "paper" ? !!it.paper : !it.paper && grp(it) === g);
    const tSel = $("#ex-type"), gSel = $("#ex-group"), lSel = $("#ex-loop"), play = $("#ex-play");
    h("option", { value: "" }, tSel, "All types");
    for (const t of S.types) h("option", { value: t.key }, tSel, t.name);
    for (const [k, v] of GROUPS) h("option", { value: k }, gSel, v);
    h("option", { value: "" }, gSel, "All");
    segment(gSel, "chips");
    let list = [], idx = 0, timer = null;
    const cut = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);
    const argmax = (p) => p.indexOf(Math.max(...p));

    function filter() {
      list = IT.items.filter((it) => (!gSel.value || inGroup(it, gSel.value)) && (!tSel.value || it.type === tSel.value));
      idx = 0; stop(); lSel.value = "1"; item();
    }
    function item() {
      const it = list[idx];
      $("#ex-count").textContent = list.length ? "Item " + (idx + 1) + " of " + list.length : "No item in this group";
      for (const id of ["#ex-tags", "#ex-opts", "#ex-one", "#ex-traj"]) $(id).textContent = "";
      $("#ex-state").textContent = it ? it.state : ""; $("#ex-question").textContent = it ? it.question : ""; $("#ex-verdict").textContent = "";
      if (!it) return;
      for (const t of [it.source, it.type_name, TIER[it.tier], LABEL[grp(it)]]) h("span", { class: "tag" }, $("#ex-tags"), t);
      $("#ex-state").scrollTop = 0;
      it.options.forEach((o, k) => {
        const row = h("div", { class: "opt" + (k === it.gold ? " is-gold" : "") }, $("#ex-opts")), txt = h("div", { class: "txt" }, row);
        h("b", {}, txt, LETTER[k] + "  "); txt.appendChild(document.createTextNode(o));
        if (k === it.gold) h("span", { class: "gold" }, txt, "correct");
        h("div", { class: "p" }, row);
        const track = h("div", { class: "track" }, row); h("div", { class: "fill" }, track);
        if (it.unans) { const thr = h("div", { class: "thr", title: "threshold of a hard answer" }, track); thr.style.left = ((1 + 1 / it.options.length) / 2) * 100 + "%"; }
      });
      /* the single-pass models */
      h("div", { class: "kicker", style: "margin-top:0" }, $("#ex-one"), "Single-pass models, one pass each");
      const tb = h("table", {}, $("#ex-one"));
      for (const [name, col] of ONE) {
        const p = it.one[name], k = argmax(p), tr = h("tr", {}, tb), m = h("td", { class: "m" }, tr), sw = h("span", { class: "swatch" }, m);
        sw.style.background = col; m.appendChild(document.createTextNode(name));
        const a = h("td", { class: "a" }, tr);
        if (it.unans) { const hard = p[k] >= (1 + 1 / p.length) / 2; a.textContent = hard ? "commits to " + LETTER[k] : "no hard answer"; }
        else { h("span", { class: "mark" }, a, k === it.gold ? "✓" : "✗"); a.appendChild(document.createTextNode(" " + LETTER[k] + "  " + cut(it.options[k], 34))); }
        h("td", { class: "pp" }, tr, f2(p[k]));
      }
      loop();
    }
    function loop() {
      const it = list[idx]; if (!it) return;
      const t = +lSel.value, p = it.p[t - 1], top = argmax(p), rows = $("#ex-opts").children;
      $("#ex-now").textContent = "Loop " + t;
      p.forEach((v, k) => {
        rows[k].classList.toggle("is-top", k === top);
        $(".p", rows[k]).textContent = f2(v);
        $(".fill", rows[k]).style.width = v * 100 + "%";
      });
      const vd = $("#ex-verdict"); vd.textContent = "";
      vd.className = "verdict" + ((it.unans ? p[top] < (1 + 1 / p.length) / 2 : top === it.gold) ? " ok" : "");
      if (it.unans) {
        const thr = (1 + 1 / p.length) / 2, hard = p[top] >= thr;
        h("b", {}, vd, hard ? "Commits to " + LETTER[top] + " (" + f2(p[top]) + "). " : "No hard answer (" + f2(p[top]) + "). ");
        vd.appendChild(document.createTextNode(hard ? "The sentence that decides the case was removed, so this is a mistake." : "The evidence is missing; staying below the threshold (" + f2(thr) + ", the mark on the bars) is the right response."));
      } else {
        h("span", { class: "mark" }, vd, top === it.gold ? "✓" : "✗");
        h("b", {}, vd, " Answers " + LETTER[top] + " (" + f2(p[top]) + ")");
        vd.appendChild(document.createTextNode(top === it.gold ? ": correct." : ": wrong; the correct option is " + LETTER[it.gold] + "."));
      }
      /* the probability of every option over the loops */
      const series = it.options.map((o, k) => ({ name: LETTER[k] + "  " + cut(o, 30), color: k === it.gold ? COL.sansi : COL.gray, width: k === it.gold ? 2 : 1.5, markers: k === it.gold, values: it.p.map((r) => r[k]) }));
      series.sort((a, b) => (a.color === COL.sansi) - (b.color === COL.sansi));
      const refs = it.unans ? [{ name: "Threshold of a hard answer", short: "hard answer", color: COL.ink2, dash: "4 3", value: (1 + 1 / p.length) / 2 }] : [];
      const items = it.unans ? [{ kind: "line", color: COL.gray, width: 1.5, label: "options (none is correct)" }, { kind: "line", color: COL.ink2, dash: "4 3", width: 1.5, label: "threshold of a hard answer" }]
        : [{ kind: "line", marker: "dot", color: COL.sansi, label: "correct option" }, { kind: "line", color: COL.gray, width: 1.5, label: "other options" }];
      lineChart($("#ex-traj"), { W: 430, H: 200, m: { t: 28, r: 12, b: 46, l: 38 }, legendItems: items, xs: LOOPS, xDomain: [0.7, 8.3], xTitle: "Loop", yDomain: [0, 1], yTicks: [0, 0.5, 1], yFmt: f2, yTitle: "Probability", series, refs, mark: t, tipTitle: (l) => "Loop " + l, aria: "Probability of every option after each loop" });
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; } }
    play.addEventListener("click", () => {
      if (timer) return stop();
      if (+lSel.value === 8) lSel.value = "1";
      loop(); play.textContent = "Pause";
      timer = setInterval(() => { if (+lSel.value >= 8) return stop(); lSel.value = String(+lSel.value + 1); loop(); }, 900);
    });
    lSel.addEventListener("input", () => { stop(); loop(); });
    $("#ex-prev").addEventListener("click", () => { if (list.length) { idx = (idx - 1 + list.length) % list.length; stop(); lSel.value = "1"; item(); } });
    $("#ex-next").addEventListener("click", () => { if (list.length) { idx = (idx + 1) % list.length; stop(); lSel.value = "1"; item(); } });
    tSel.addEventListener("change", filter); gSel.addEventListener("change", filter);
    const sh = S.dynamics.shape;
    const nPaper = IT.items.filter((it) => it.paper).length;
    $("#ex-note").textContent = "All items come from public datasets and are shown with the probabilities of training seed 0: the " + nPaper + " examples of the paper, and " + (IT.items.length - nPaper) +
      " items drawn at random within each group (item type and what the loops did). " +
      "The random groups appear in similar numbers here, not in their real proportions. Of all answerable test items, " + f1(sh.always_right) + "% are right at every loop, " + f1(sh.fixed_once) + "% are fixed once and stay right, " +
      f1(sh.broken_once) + "% are broken once and stay wrong, " + f1(sh.back_and_forth_right + sh.back_and_forth_wrong) + "% move back and forth, and " + f1(sh.always_wrong_same + sh.always_wrong_moving) + "% are wrong at every loop.";
    filter();
  })();
}).catch((err) => {
  const box = document.createElement("p");
  box.className = "note wrap";
  box.textContent = "The data of this page could not be loaded (" + err.message + "). Open it through a web server, not as a local file.";
  document.querySelector("main").prepend(box);
});
})();
