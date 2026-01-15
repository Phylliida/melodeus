document.addEventListener("DOMContentLoaded", () => {
  const live = $("transcript-live");
  const log = $("transcript-log");
  const entries = new Map();

  const render = (data) => {
    console.log(data);
    if (!data || !data.text) return;
    if (!data.is_final) {
      if (live) live.textContent = data.text;
      return;
    }
    if (live) live.textContent = "";
    const key = data.message_id || data.text;
    let row = entries.get(key);
    if (!row) {
      row = document.createElement("div");
      row.className = "pill";
      log?.append(row);
      entries.set(key, row);
    }
    row.textContent = data.text;
  };

  if (window.registerTranscriptHandler) {
    window.registerTranscriptHandler(render);
  }
});
