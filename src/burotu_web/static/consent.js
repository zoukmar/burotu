// Cookie consent, then Google Tag Manager (GA4). Nothing is loaded and nothing is sent to Google
// until the visitor clicks "Accept". The choice is kept in this browser for 6 months.
(function () {
  const GTM_ID = "GTM-P5JNNMNJ";
  const KEY = "burotu-consent";
  const MAX_AGE = 182 * 24 * 3600 * 1000;
  const banner = document.getElementById("consent");

  function read() {
    try {
      const v = JSON.parse(localStorage.getItem(KEY) || "null");
      return v && Date.now() - v.at < MAX_AGE ? v.choice : null;
    } catch (e) { return null; }
  }
  function save(choice) {
    try { localStorage.setItem(KEY, JSON.stringify({ choice, at: Date.now() })); } catch (e) { /* private mode */ }
  }
  let loaded = false;
  function loadGtm() {
    if (loaded) return;
    loaded = true;
    window.dataLayer = window.dataLayer || [];
    function gtag() { window.dataLayer.push(arguments); }
    gtag("consent", "default", { ad_storage: "denied", ad_user_data: "denied", ad_personalization: "denied", analytics_storage: "granted" });
    window.dataLayer.push({ "gtm.start": Date.now(), event: "gtm.js" });
    const s = document.createElement("script");
    s.async = true;
    s.src = "https://www.googletagmanager.com/gtm.js?id=" + GTM_ID;
    document.head.appendChild(s);
  }
  function show() { if (banner) banner.hidden = false; }
  function hide() { if (banner) banner.hidden = true; }

  const choice = read();
  if (choice === "granted") loadGtm();
  else if (choice === null) show();

  if (banner) {
    banner.querySelector("[data-consent=accept]").addEventListener("click", () => { save("granted"); hide(); loadGtm(); });
    banner.querySelector("[data-consent=refuse]").addEventListener("click", () => { save("denied"); hide(); });
  }
  document.querySelectorAll("[data-consent-open]").forEach((a) =>
    a.addEventListener("click", (e) => { e.preventDefault(); show(); }));
})();
