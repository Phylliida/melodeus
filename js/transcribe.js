document.addEventListener("DOMContentLoaded", () => {
  const live = $("transcript-live");
  const log = $("transcript-log");
  const entries = new Map();

  const render = (data) => {
    if (!data || !data.text || !log) return;
    const key = data.message_id || data.text;
    let bubble = entries.get(key);
    if (!bubble) {
      bubble = document.createElement("div");
      bubble.className = "pill";
      bubble.dataset.messageId = key;
      log.append(bubble);
      entries.set(key, bubble);
    }
    bubble.textContent = data.text;
    if (live) live.textContent = "";
  };

  if (window.registerTranscriptHandler) {
    window.registerTranscriptHandler(render);
  }
});
