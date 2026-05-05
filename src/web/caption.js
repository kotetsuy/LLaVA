// Caption: subscribes to /ws/caption and shows the latest VLM output.
// Updates roughly every 2 s (the VLM cadence); we cross-fade between captions
// for a calmer look since the text is reasonably long.

(() => {
  const $caption = document.getElementById('caption');
  const $status = document.getElementById('caption-status');

  function setCaption(text, timing) {
    if (!text) text = '(no caption yet)';
    $caption.classList.add('fading');
    setTimeout(() => {
      $caption.textContent = text;
      $caption.classList.remove('fading');
    }, 120);
    if (timing && $status) {
      const inf = timing.inference_ms != null ? `${Math.round(timing.inference_ms)}ms` : '?';
      const tps = timing.eval_tps != null ? `${timing.eval_tps.toFixed(1)} t/s` : '';
      $status.textContent = `caption: ${inf} ${tps}`.trim();
    }
  }

  let reconnectDelay = 500;

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const ws = new WebSocket(proto + location.host + '/ws/caption');

    ws.onopen = () => {
      reconnectDelay = 500;
      if ($status) $status.textContent = 'caption: connected';
    };
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        setCaption(msg.caption, msg.timing);
      } catch (err) {
        console.error('caption parse error', err);
      }
    };
    ws.onclose = () => {
      if ($status) $status.textContent = 'caption: reconnecting…';
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 5000);
    };
  }

  connect();
})();
