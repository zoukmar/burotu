// Réservation : charge les horaires libres du centre et du jour choisis.
(function () {
  const done = document.getElementById("booking-done");
  if (done) done.focus();

  const form = document.getElementById("book-form");
  if (!form) return;
  const day = document.getElementById("day");
  const box = document.getElementById("slots");
  let selected = box.dataset.selected || "";

  function message(text) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = text;
    box.replaceChildren(p);
  }

  async function load() {
    const site = form.querySelector('input[name="site_id"]:checked');
    if (!site || !day.value) return message(form.dataset.msgPick);
    const url = `/api/slots?site=${encodeURIComponent(site.value)}&day=${encodeURIComponent(day.value)}`;
    let slots = [];
    try {
      const res = await fetch(url);
      if (res.ok) slots = (await res.json()).slots;
    } catch (_) {
      /* réseau indisponible : on garde le message par défaut */
    }
    if (!slots.length) return message(form.dataset.msgNone);
    box.replaceChildren(
      ...slots.map((s) => {
        const label = document.createElement("label");
        label.className = "slot";
        const input = Object.assign(document.createElement("input"), {
          type: "radio", name: "slot", value: s, required: true, checked: s === selected,
        });
        input.addEventListener("change", () => { selected = s; });
        const span = document.createElement("span");
        span.textContent = s;
        label.append(input, span);
        return label;
      })
    );
  }

  form.addEventListener("change", (e) => {
    if (e.target.name === "site_id" || e.target === day) load();
  });
  load();

  const errors = document.getElementById("errors");
  if (errors) errors.focus();
})();
