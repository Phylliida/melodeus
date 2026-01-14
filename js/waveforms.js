

const set_ws_status_color = (c) => (document.getElementById("ws").style.background = c)

const wsPort = 5001;
const url = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.hostname}:${wsPort}`;

const connect = () => {
  const ws = new WebSocket(url());
  ws.onopen = () => set_ws_status_color("#3f3");
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      console.log("Got data");
    } catch (e) {}
    set_ws_status_color("#3f3");
  }
  const stop = () => {
    set_ws_status_color("#f33");
    connect();
  }
  ws.onclose = stop;
  ws.onerror = stop;
};

document.addEventListener("DOMContentLoaded", () => {
    set_ws_status_color("#ff0");
    connect();
});