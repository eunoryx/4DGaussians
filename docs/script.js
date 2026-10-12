(function () {
  "use strict";

  var DATA = window.SITE_DATA || { palette: [], blobs: [], aspect: 4 / 3 };
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  var box = document.getElementById("compare");
  var range = document.getElementById("compare-range");
  var ours = document.getElementById("vid-ours");
  var base = document.getElementById("vid-base");
  var canvas = document.getElementById("blobs");
  if (!box || !range) return;

  /* ---------- Slider: a native range input, so drag, touch and arrow keys all work ---------- */

  function setPos() { box.style.setProperty("--pos", range.value + "%"); }
  range.addEventListener("input", setPos);
  setPos();

  /* ---------- Videos: loaded only when needed, kept on the same frame ---------- */

  var START = 3.5;            // the posters and the blobs are this frame of the clip
  var loaded = false, inView = true, settled = false;

  function playBoth() {
    if (reduceMotion.matches || !loaded || !inView || !settled) return;
    [ours, base].forEach(function (v) { var p = v.play(); if (p && p.catch) p.catch(function () {}); });
  }

  function loadVideos() {
    if (loaded || !ours || !base) return;
    loaded = true;
    [ours, base].forEach(function (v) {
      v.addEventListener("loadedmetadata", function () { v.currentTime = START; }, { once: true });
      v.preload = "auto";
      v.src = v.dataset.src;
      v.load();
    });
    ours.addEventListener("timeupdate", function () {
      if (Math.abs(base.currentTime - ours.currentTime) > 0.07) base.currentTime = ours.currentTime;
    });
  }

  if (ours && base && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      inView = entries[0].isIntersecting;
      if (inView) playBoth(); else if (loaded) { ours.pause(); base.pause(); }
    }).observe(box);
  }

  function afterSettle() {
    settled = true;
    if (canvas) canvas.parentNode.hidden = true;
    if (reduceMotion.matches) return;      // the two still frames stay; nothing moves
    loadVideos();
    if (ours.readyState >= 2) playBoth(); else ours.addEventListener("canplay", playBoth, { once: true });
  }

  /* ---------- On load: translucent Gaussian blobs settle into the render ---------- */

  if (!canvas || !DATA.blobs.length || reduceMotion.matches) { afterSettle(); return; }

  var ctx = canvas.getContext("2d");
  var FIELDS = 9, ROOT = 125, SETTLE = 3.9, FADE = 0.7;

  var sprites = DATA.palette.map(function (hex) {
    var s = document.createElement("canvas");
    s.width = s.height = 128;
    var c = s.getContext("2d");
    var g = c.createRadialGradient(64, 64, 0, 64, 64, 64);
    var rgb = [1, 3, 5].map(function (i) { return parseInt(hex.slice(i, i + 2), 16); }).join(",");
    for (var i = 0; i <= 20; i++) {
      var r = i / 20;
      g.addColorStop(r, "rgba(" + rgb + "," + (Math.exp(-r * r * 4.6) * (1 - r * r)).toFixed(3) + ")");
    }
    c.fillStyle = g;
    c.fillRect(0, 0, 128, 128);
    return s;
  });

  var seed = 11;
  function rand() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }

  var blobs = [], d = DATA.blobs;
  for (var i = 0; i < d.length; i += FIELDS) {
    var small = 1 - Math.min(1, d[i + 2] / ROOT);
    blobs.push({
      x: d[i] / 1000, y: d[i + 1] / 1000, s: d[i + 2] / 1000, asp: d[i + 3] / 100, rot: d[i + 4] / 100, k: d[i + 5],
      px: d[i + 6] / 1000, py: d[i + 7] / 1000, moves: d[i + 8] > 1,
      a: 0.5 + 0.3 * small, a0: (0.5 + 0.3 * small) / Math.max(1, Math.sqrt(d[i + 8])),
      delay: 0.35 + small * 1.5 + rand() * 0.3
    });
  }
  blobs.sort(function (p, q) { return q.s - p.s; });

  var strays = [];
  for (var j = 0; j < 70; j++) {
    strays.push({ x: rand(), y: rand(), s: 0.008 + rand() * 0.016, k: Math.floor(rand() * sprites.length), out: 2.2 + rand() * 1.2 });
  }

  var W = 0, H = 0, dpr = 1;
  function resize() {
    var r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }

  function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

  function sprite(k, x, y, r, asp, rot, alpha) {
    if (alpha <= 0.004) return;
    var c = Math.cos(rot) * dpr, s = Math.sin(rot) * dpr;
    ctx.setTransform(c, s, -s, c, x * dpr, y * dpr);
    ctx.globalAlpha = alpha;
    ctx.drawImage(sprites[k], -r, -r * asp, 2 * r, 2 * r * asp);
  }

  function draw(t) {
    var out = Math.min(1, Math.max(0, (t - SETTLE) / FADE));    // the whole layer lifts off the real render
    canvas.parentNode.style.opacity = 1 - out;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#fff";                                      // white multiplies to the paper colour
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    var fadeIn = Math.min(1, t / 0.5);
    var rootR = ROOT / 1000 * W * 1.15;

    for (var n = 0; n < strays.length; n++) {
      var f = strays[n], gone = Math.min(1, Math.max(0, (t - f.out) / 0.6));
      sprite(f.k, f.x * W, f.y * H, f.s * W * (1 - 0.4 * gone), 1, 0, 0.32 * fadeIn * (1 - gone));
    }
    for (var i = 0; i < blobs.length; i++) {
      var b = blobs[i];
      var e = b.moves ? ease(Math.min(1, Math.max(0, (t - b.delay) / 1.4))) : 1;
      var r = b.moves ? rootR + (b.s * W * 2.3 - rootR) * e : b.s * W * 2.3;
      sprite(b.k, (b.px + (b.x - b.px) * e) * W, (b.py + (b.y - b.py) * e) * H, r, 1 + (b.asp - 1) * e, b.rot, (b.a0 + (b.a - b.a0) * e) * fadeIn);
    }
  }

  var start = 0;
  function frame(now) {
    if (!start) start = now;
    var t = (now - start) / 1000;
    if (t >= SETTLE + FADE) { afterSettle(); return; }
    draw(t);
    requestAnimationFrame(frame);
  }

  resize();
  window.addEventListener("resize", function () { if (!settled) resize(); });
  loadVideos();                        // fetch while the blobs settle, so the render is ready underneath
  requestAnimationFrame(frame);
})();
