// Booking: load free slots for the chosen site and day, and point to the next free day.
(function () {
  const done = document.getElementById("booking-done");
  if (done) done.focus();

  const form = document.getElementById("book-form");
  if (!form) return;
  const day = document.getElementById("day");
  const box = document.getElementById("slots");
  let selected = box.dataset.selected || "";
  let dayChosenByPerson = Boolean(day.value);

  function message(text, nextDay) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = text;
    const parts = [p];
    if (nextDay) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn btn-small btn-ghost";
      const label = new Date(nextDay + "T12:00:00").toLocaleDateString(form.dataset.lang, {
        weekday: "long", day: "numeric", month: "long",
      });
      b.textContent = `${form.dataset.msgNext} ${label}`;
      b.addEventListener("click", () => { day.value = nextDay; load(); });
      parts.push(b);
    }
    box.replaceChildren(...parts);
  }

  async function load() {
    const site = form.querySelector('input[name="site_id"]:checked');
    if (!site) return message(form.dataset.msgPick);
    if (!day.value) day.value = day.min;  // start from today, then jump to the first free day
    const url = `/api/slots?site=${encodeURIComponent(site.value)}&day=${encodeURIComponent(day.value)}`;
    let data = { slots: [], next_day: null };
    try {
      const res = await fetch(url);
      if (res.ok) data = await res.json();
    } catch (_) {
      /* network unavailable: keep the default message */
    }
    if (!data.slots.length) {
      if (!dayChosenByPerson && data.next_day) {
        day.value = data.next_day;  // nobody picked a date yet: show the first one with free times
        return load();
      }
      return message(form.dataset.msgNone, data.next_day);
    }
    box.replaceChildren(
      ...data.slots.map((s) => {
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
    if (e.target === day) dayChosenByPerson = true;
    if (e.target.name === "site_id" || e.target === day) load();
  });
  load();

  const errors = document.getElementById("errors");
  if (errors) errors.focus();
})();
