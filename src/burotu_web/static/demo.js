// Live demo: run one demo lab file through every model and reveal the results step by step.
(function () {
  const picker = document.getElementById("demo-picker");
  const out = document.getElementById("demo-out");
  if (!picker || !out) return;
  const lang = picker.dataset.lang;
  const T = JSON.parse(document.getElementById("demo-text").textContent);
  const SVG = "http://www.w3.org/2000/svg";
  const pct = (x, d = 0) => (x * 100).toLocaleString(lang, { maximumFractionDigits: d, minimumFractionDigits: d }) + " %";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function bar(value, cls) {
    const s = document.createElementNS(SVG, "svg");
    s.setAttribute("viewBox", "0 0 100 8");
    s.setAttribute("preserveAspectRatio", "none");
    s.setAttribute("class", "demo-bar " + (cls || ""));
    const track = document.createElementNS(SVG, "rect");
    track.setAttribute("width", "100"); track.setAttribute("height", "8"); track.setAttribute("rx", "4"); track.setAttribute("class", "track");
    const fill = document.createElementNS(SVG, "rect");
    fill.setAttribute("width", String(Math.max(1, Math.min(100, value * 100)).toFixed(1)));
    fill.setAttribute("height", "8"); fill.setAttribute("rx", "4"); fill.setAttribute("class", "fill");
    s.append(track, fill);
    return s;
  }
  function stage(title, sub) {
    const box = el("div", "demo-stage is-wait");
    const head = el("div", "demo-stage-head");
    head.append(el("span", "demo-dot"), el("strong", "", title));
    if (sub) head.append(el("span", "small muted", sub));
    const body = el("div", "demo-stage-body");
    box.append(head, body);
    return { box, body, done(ok) { box.classList.remove("is-wait"); box.classList.add(ok === false ? "is-ko" : "is-ok"); } };
  }
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const follow = (node) => node.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "center" });
  const tissue = (data, c) => (data.tissues && data.tissues[c]) || c;
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function modelBody(data, key, body) {
    const m = data.models[key];
    if (!m) { body.append(el("p", "small muted", T.meta_off)); return false; }
    const row = el("div", "demo-model");
    const verdict = el("span", "badge " + (m.signal ? "is-signal" : "is-clear"), m.signal ? T.signal : T.no_signal);
    row.append(verdict);
    const list = el("ol", "demo-origins");
    m.origin.forEach((o) => {
      const li = el("li");
      li.append(el("span", "", cap(tissue(data, o.class))), bar(o.probability), el("span", "num small", pct(o.probability)));
      list.append(li);
    });
    row.append(list);
    body.append(row);
    return true;
  }

  async function run(sample, button) {
    picker.querySelectorAll("button").forEach((b) => { b.disabled = true; b.classList.toggle("is-picked", b === button); });
    out.replaceChildren();
    const stages = {
      qc: stage(T.qc),
      classic: stage(T.classic, T.classic_d),
      logreg: stage(T.logreg, T.logreg_d),
      meta: stage(T.meta, T.meta_d),
      consensus: stage(T.consensus, T.consensus_d),
    };
    stages.meta.box.classList.add("is-meta");
    const grid = el("div", "demo-stages");
    Object.values(stages).forEach((s) => grid.append(s.box));
    const status = el("p", "small muted demo-status", T.running);
    out.append(status, grid);
    out.scrollIntoView({ behavior: "smooth", block: "start" });

    let data;
    try {
      const r = await fetch(`/api/demo/${encodeURIComponent(sample)}?lang=${lang}`, { method: "POST" });
      if (r.status === 429) throw new Error("limit");
      if (!r.ok) throw new Error("error");
      data = await r.json();
    } catch (e) {
      status.textContent = e.message === "limit" ? T.limit : T.error;
      picker.querySelectorAll("button").forEach((b) => { b.disabled = false; });
      return;
    }
    const q = data.quality;
    status.textContent = data.cached_min !== undefined
      ? `${T.time.replace("{s}", data.elapsed.toLocaleString(lang))} · ${T.cached.replace("{m}", data.cached_min)}`
      : T.time.replace("{s}", data.elapsed.toLocaleString(lang));

    await sleep(300);
    const qcText = { pass: T.qc_pass, warn: T.qc_warn, fail: T.qc_fail }[q.status];
    stages.qc.body.append(el("p", "demo-big", qcText),
      el("p", "small muted", `${pct(q.coverage, 1)} ${T.coverage} · ${Math.round(q.depth)} ${T.depth}`));
    stages.qc.done(q.status !== "fail");
    if (q.status === "fail") {
      ["classic", "logreg", "meta", "consensus"].forEach((k) => stages[k].box.classList.add("is-skip"));
    } else {
      for (const key of ["classic", "logreg", "meta"]) {
        await sleep(key === "meta" ? 700 : 400);
        const ok = modelBody(data, key, stages[key].body);
        stages[key].done(ok ? undefined : false);
      }
      await sleep(400);
      const cs = data.consensus;
      if (cs) {
        stages.consensus.body.append(
          el("p", "demo-big", cap(tissue(data, cs.tissue))),
          el("p", "small muted", `${T.signal}: ${cs.signal_votes} ${T.of} ${cs.n} · ${T.tissue}: ${cs.tissue_votes} ${T.of} ${cs.n}`));
      }
      stages.consensus.done();

      if (data.explain) {
        await sleep(400);
        const x = data.explain;
        const why = el("div", "demo-why");
        why.append(el("h3", "", T.why));
        const g = el("div", "demo-why-grid");
        const a = el("div", "demo-card");
        a.append(el("p", "small muted", T.rank), el("p", "demo-big", pct(x.rank, 2)), el("p", "small", T.rank_d));
        const b = el("div", "demo-card");
        b.append(el("p", "small muted", T.lit), el("p", "demo-big", `${x.lit} / ${x.sites.toLocaleString(lang)}`),
          el("p", "small", `${T.lit_d} ${x.healthy_p99}.`));
        const c = el("div", "demo-card");
        c.append(el("p", "small muted", T.tissue));
        const ol = el("ol", "demo-origins");
        x.tissues.forEach((t) => {
          const li = el("li");
          li.append(el("span", "", cap(tissue(data, t.class))), bar(t.lit / Math.max(1, t.markers), "is-lit"), el("span", "num small", `${t.lit}/${t.markers}`));
          ol.append(li);
        });
        c.append(ol);
        g.append(a, b, c);
        why.append(g);
        out.append(why);
        follow(why);
      }
    }

    await sleep(500);
    if (data.report) {
      const box = el("div", "demo-letter");
      const head = el("div", "demo-letter-head");
      head.append(el("h3", "", T.letter));
      const quote = el("blockquote", "", data.report.text);
      quote.lang = lang;
      box.append(head, quote, el("p", "small muted", data.report.spark ? T.by_spark : T.by_template));
      out.append(box);
      follow(box);
    }

    await sleep(600);
    const truth = data.truth;
    const card = el("div", "demo-truth");
    card.append(el("p", "kicker", T.truth));
    let text;
    if (truth.label === "Healthy") text = T.truth_healthy;
    else text = T.truth_tumour.replace("{tissue}", tissue(data, truth.label)).replace("{tf}", pct(truth.tf, truth.tf < 0.01 ? 1 : 0));
    card.append(el("p", "demo-big", text));
    if (truth.depth < 100) card.append(el("p", "", T.truth_shallow.replace("{depth}", truth.depth)));
    out.append(card);
    follow(card);
    picker.querySelectorAll("button").forEach((b) => { b.disabled = false; });
  }

  picker.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-sample]");
    if (b) run(b.dataset.sample, b);
  });
})();
