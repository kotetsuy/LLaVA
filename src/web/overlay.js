// Bbox overlay: connects to /ws/bbox and draws detection boxes on a canvas
// sized to the SHM frame (1280x720). The CSS layout stretches both video and
// canvas to fit the same .stage rect, so canvas pixel coordinates align with
// video display coordinates.

(() => {
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const wsStatus = document.getElementById('ws-status');

  // Default to the SHM normalized frame size; the first message tells us if
  // the actual size differs (e.g. a future config change).
  canvas.width = 1280;
  canvas.height = 720;

  const COLORS = {
    person: 'rgb(50, 220, 80)',
    car:    'rgb(255, 200, 60)',
    dog:    'rgb(255, 120, 200)',
    cat:    'rgb(120, 200, 255)',
  };
  const DEFAULT_COLOR = 'rgb(80, 180, 255)';

  function colorFor(label) {
    return COLORS[label] || DEFAULT_COLOR;
  }

  function drawBoxes(msg) {
    if (msg.frame_w !== canvas.width || msg.frame_h !== canvas.height) {
      canvas.width = msg.frame_w;
      canvas.height = msg.frame_h;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = '14px -apple-system, "Helvetica Neue", system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';

    for (const b of msg.boxes) {
      const color = colorFor(b.label);
      const w = b.x2 - b.x1;
      const h = b.y2 - b.y1;

      ctx.strokeStyle = color;
      ctx.strokeRect(b.x1, b.y1, w, h);

      const text = `${b.label} ${(b.conf * 100).toFixed(0)}%`;
      const tw = ctx.measureText(text).width + 8;
      const labelTop = b.y1 < 18 ? b.y1 : b.y1 - 18;

      ctx.fillStyle = color;
      ctx.fillRect(b.x1, labelTop, tw, 18);
      ctx.fillStyle = '#000';
      ctx.fillText(text, b.x1 + 4, labelTop + 14);
    }
  }

  let reconnectDelay = 500;

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const ws = new WebSocket(proto + location.host + '/ws/bbox');

    ws.onopen = () => {
      reconnectDelay = 500;
      if (wsStatus) wsStatus.textContent = 'bbox: connected';
    };
    ws.onmessage = (e) => {
      try {
        drawBoxes(JSON.parse(e.data));
      } catch (err) {
        console.error('bbox parse/draw error', err);
      }
    };
    ws.onclose = () => {
      if (wsStatus) wsStatus.textContent = 'bbox: reconnecting…';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 5000);  // cap at 5s
    };
    ws.onerror = () => {
      // onclose will fire too; let it handle reconnect.
    };
  }

  connect();
})();
