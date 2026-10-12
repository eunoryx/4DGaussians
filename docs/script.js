(function () {
  "use strict";

  var DATA = window.SITE_DATA || { palette: [], blobs: [], aspect: 4 / 3, countBaseline: [], countOurs: [] };
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ------------------------------------------------------------------
     Hero: Gaussian blobs that start coarse, split toward the complex
     parts of a real render frame, and settle. Strays are pruned.
     ------------------------------------------------------------------ */
  (function hero() {
    var canvas = document.getElementById("hero-canvas");
    if (!canvas || !DATA.blobs.length) return;
    var ctx = canvas.getContext("2d");
    var replay = document.getElementById("hero-replay");
    var FIELDS = 9, ROOT_SIZE = 125; // root cell is 90px of a 720px frame, in per-mille of width
    var SETTLE = 4.8;

    // One soft sprite per palette colour; drawing a sprite is much cheaper than a gradient per blob.
    var sprites = DATA.palette.map(function (hex) {
      var s = document.createElement("canvas");
      s.width = s.height = 48;
      var c = s.getContext("2d");
      var g = c.createRadialGradient(24, 24, 0, 24, 24, 24);
      var rgb = [1, 3, 5].map(function (i) { return parseInt(hex.slice(i, i + 2), 16); }).join(",");
      for (var i = 0; i <= 8; i++) {
        var r = i / 8;
        g.addColorStop(r, "rgba(" + rgb + "," + (Math.exp(-r * r * 4.6) * (1 - r * r)).toFixed(3) + ")");
      }
      c.fillStyle = g;
      c.fillRect(0, 0, 48, 48);
      return s;
    });

    var seed = 11;
    function rand() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }

    var blobs = [];
    for (var i = 0; i < DATA.blobs.length; i += FIELDS) {
      var d = DATA.blobs, s = d[i + 2];
      var small = 1 - Math.min(1, s / ROOT_SIZE);
      blobs.push({
        x: d[i] / 1000, y: d[i + 1] / 1000, s: s / 1000, asp: d[i + 3] / 100, rot: d[i + 4] / 100,
        k: d[i + 5], px: d[i + 6] / 1000, py: d[i + 7] / 1000,
        a: 0.5 + 0.3 * small, a0: (0.5 + 0.3 * small) / Math.max(1, Math.sqrt(d[i + 8])),
        delay: 0.5 + small * 1.7 + rand() * 0.35, phase: rand() * 6.283, moves: d[i + 8] > 1
      });
    }
    blobs.sort(function (p, q) { return q.s - p.s; });

    var strays = [];
    for (var j = 0; j < 80; j++) {
      strays.push({ x: -0.05 + rand() * 1.1, y: -0.12 + rand() * 1.2, s: 0.008 + rand() * 0.018,
        k: Math.floor(rand() * sprites.length), out: 2.9 + rand() * 1.5, phase: rand() * 6.283 });
    }

    var W = 0, H = 0, dpr = 1, scene = { x: 0, y: 0, w: 0, h: 0 };

    function resize() {
      var rect = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = rect.width; H = rect.height;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      if (W > 820) {
        scene.w = Math.min(W * 0.47, H * 0.62 * DATA.aspect);
        scene.h = scene.w / DATA.aspect;
        scene.x = W - scene.w - Math.max(16, W * 0.03);
        scene.y = Math.max(20, H * 0.05);
      } else {
        scene.w = Math.min(W, 600);
        scene.h = scene.w / DATA.aspect;
        scene.x = (W - scene.w) / 2;
        scene.y = 20;
      }
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
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var fadeIn = Math.min(1, t / 0.6);
      var idle = Math.min(1, Math.max(0, (t - SETTLE) / 1.5));
      var rootR = ROOT_SIZE / 1000 * scene.w * 1.15;

      for (var n = 0; n < strays.length; n++) {
        var f = strays[n];
        var gone = Math.min(1, Math.max(0, (t - f.out) / 0.7));
        sprite(f.k, scene.x + f.x * scene.w + Math.sin(t * 0.5 + f.phase) * 3, scene.y + f.y * scene.h + Math.cos(t * 0.4 + f.phase) * 3,
          f.s * scene.w * (1 - 0.4 * gone), 1, 0, 0.3 * fadeIn * (1 - gone));
      }

      for (var i = 0; i < blobs.length; i++) {
        var b = blobs[i];
        var e = b.moves ? ease(Math.min(1, Math.max(0, (t - b.delay) / 1.5))) : 1;
        var x = scene.x + (b.px + (b.x - b.px) * e) * scene.w;
        var y = scene.y + (b.py + (b.y - b.py) * e) * scene.h;
        var r = b.moves ? rootR + (b.s * scene.w * 2.3 - rootR) * e : b.s * scene.w * 2.3;
        if (idle) {
          x += Math.sin(t * 0.55 + b.phase) * 0.7 * idle;
          y += Math.cos(t * 0.45 + b.phase) * 0.7 * idle;
        }
        sprite(b.k, x, y, r, 1 + (b.asp - 1) * e, b.rot, (b.a0 + (b.a - b.a0) * e) * fadeIn);
      }
    }

    var start = 0, raf = 0, visible = true;

    function frame(now) {
      raf = 0;
      if (!start) start = now;
      draw((now - start) / 1000);
      if (visible && !document.hidden) raf = requestAnimationFrame(frame);
    }

    function run() {
      cancelAnimationFrame(raf);
      raf = 0;
      if (reduceMotion.matches) {
        draw(SETTLE);               // the settled frame, with no motion at all
        if (replay) replay.hidden = true;
        return;
      }
      if (replay) replay.hidden = false;
      raf = requestAnimationFrame(frame);
    }

    resize();
    run();

    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { resize(); if (reduceMotion.matches || !raf) draw(reduceMotion.matches ? SETTLE : (performance.now() - start) / 1000); }, 80);
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        visible = entries[0].isIntersecting;
        if (visible && !raf && !reduceMotion.matches) raf = requestAnimationFrame(frame);
      }).observe(canvas);
    }
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && visible && !raf && !reduceMotion.matches) raf = requestAnimationFrame(frame);
    });
    if (reduceMotion.addEventListener) reduceMotion.addEventListener("change", function () { start = 0; run(); });
    if (replay) replay.addEventListener("click", function () { start = 0; run(); });
  })();

  /* ------------------------------------------------------------------
     Before / after: two synced videos, a native range input on top
     so dragging, touch and arrow keys all work.
     ------------------------------------------------------------------ */
  (function compare() {
    var box = document.getElementById("compare");
    var range = document.getElementById("compare-range");
    var ours = document.getElementById("vid-ours");
    var base = document.getElementById("vid-base");
    if (!box || !range) return;

    function setPos() { box.style.setProperty("--pos", range.value + "%"); }
    range.addEventListener("input", setPos);
    setPos();

    if (!ours || !base) return;
    var loaded = false, inView = false;

    function play() {
      if (reduceMotion.matches) return;      // posters stay; the still frames are still comparable
      [ours, base].forEach(function (v) { var p = v.play(); if (p && p.catch) p.catch(function () {}); });
    }
    function load() {
      if (loaded) return;
      loaded = true;
      [ours, base].forEach(function (v) { v.src = v.dataset.src; v.load(); });
    }
    // Keep the two renders on the same frame.
    ours.addEventListener("timeupdate", function () {
      if (Math.abs(base.currentTime - ours.currentTime) > 0.07) base.currentTime = ours.currentTime;
    });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        inView = entries[0].isIntersecting;
        if (inView) { load(); play(); } else if (loaded) { ours.pause(); base.pause(); }
      }, { rootMargin: "200px 0px" }).observe(box);
    } else { load(); play(); }
  })();

  /* ------------------------------------------------------------------
     Gaussian count vs training step, drawn as SVG at the real pixel
     width so labels stay legible on a phone.
     ------------------------------------------------------------------ */
  (function plot() {
    var host = document.getElementById("plot");
    var readout = document.getElementById("plot-readout");
    var A = DATA.countBaseline, B = DATA.countOurs;
    if (!host || !A.length) return;
    var NS = "http://www.w3.org/2000/svg";
    var XMAX = 23000, YMAX = 30000;
    var fmt = function (n) { return n.toLocaleString("en-US"); };

    function el(name, attrs, text) {
      var e = document.createElementNS(NS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      if (text != null) e.textContent = text;
      return e;
    }

    function render() {
      var w = Math.max(280, host.clientWidth);
      var narrow = w < 560;
      var h = Math.round(Math.min(460, Math.max(260, w * 0.5)));
      var m = { l: narrow ? 34 : 46, r: 10, t: 26, b: 34 };
      var pw = w - m.l - m.r, ph = h - m.t - m.b;
      var X = function (v) { return m.l + v / XMAX * pw; };
      var Y = function (v) { return m.t + ph - v / YMAX * ph; };
      var svg = el("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h, "aria-hidden": "true" });

      [0, 10000, 20000, 30000].forEach(function (v) {
        svg.appendChild(el("line", { class: "grid", x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v) }));
        svg.appendChild(el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, v ? v / 1000 + "k" : "0"));
      });
      (narrow ? [0, 8000, 18000] : [0, 3000, 8000, 13000, 18000, 23000]).forEach(function (v) {
        svg.appendChild(el("text", { x: X(v), y: h - 10, "text-anchor": v === 0 ? "start" : v === XMAX ? "end" : "middle" }, v ? v / 1000 + "k steps" : "0"));
      });
      [[3000, narrow ? "fine stage" : "fine stage begins"], [18000, narrow ? "densify ends" : "densification ends"]].forEach(function (mk) {
        svg.appendChild(el("line", { class: "mark", x1: X(mk[0]), x2: X(mk[0]), y1: m.t - 6, y2: m.t + ph }));
        svg.appendChild(el("text", { x: X(mk[0]) + 6, y: m.t + ph - 8 }, mk[1]));
      });

      function path(series) {
        return series.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join("");
      }
      svg.appendChild(el("path", { class: "line line--base", d: path(A) }));
      svg.appendChild(el("path", { class: "line line--ours", d: path(B) }));
      var la = A[A.length - 1], lb = B[B.length - 1];
      svg.appendChild(el("text", { class: "end--base", x: X(la[0]) - 2, y: Y(la[1]) - 10, "text-anchor": "end" }, fmt(la[1])));
      svg.appendChild(el("text", { class: "end--ours", x: X(lb[0]) - 2, y: Y(lb[1]) + 20, "text-anchor": "end" }, fmt(lb[1])));

      var cursor = el("line", { class: "cursor", y1: m.t, y2: m.t + ph, visibility: "hidden" });
      svg.appendChild(cursor);

      function move(ev) {
        var box = svg.getBoundingClientRect();
        var step = ((ev.clientX - box.left) / box.width * w - m.l) / pw * XMAX;
        var i = Math.max(0, Math.min(A.length - 1, Math.round(step / 100)));
        cursor.setAttribute("x1", X(A[i][0])); cursor.setAttribute("x2", X(A[i][0]));
        cursor.setAttribute("visibility", "visible");
        if (readout) readout.textContent = "Step " + fmt(A[i][0]) + ": baseline " + fmt(A[i][1]) + ", ours " + fmt(B[Math.min(i, B.length - 1)][1]);
      }
      svg.addEventListener("pointermove", move);
      svg.addEventListener("pointerdown", move);
      svg.addEventListener("pointerleave", function () { cursor.setAttribute("visibility", "hidden"); if (readout) readout.textContent = ""; });

      host.textContent = "";
      host.appendChild(svg);
    }

    render();
    var lastW = host.clientWidth, timer;
    window.addEventListener("resize", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { if (host.clientWidth !== lastW) { lastW = host.clientWidth; render(); } }, 120);
    });
  })();
})();
