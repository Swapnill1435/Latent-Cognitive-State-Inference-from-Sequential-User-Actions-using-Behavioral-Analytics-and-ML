/**
 * WebSocket client with auto-reconnect and event queuing.
 * Sends telemetry events and receives cognitive state predictions.
 */

export class CognitiveWebSocket {
  constructor(sessionId, onMessage) {
    this.sessionId = sessionId;
    this.onMessage = onMessage;
    this.ws = null;
    this.queue = [];
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1000;
    this.onConnectionChange = null;
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/ws/${this.sessionId}`;

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.error('WebSocket connection failed:', err);
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log('🔌 WebSocket connected');
      this.connected = true;
      this.reconnectAttempts = 0;
      if (this.onConnectionChange) this.onConnectionChange(true);

      // Flush queued events
      while (this.queue.length > 0) {
        const msg = this.queue.shift();
        this.ws.send(JSON.stringify(msg));
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (this.onMessage) this.onMessage(data);
      } catch (err) {
        console.error('WebSocket message parse error:', err);
      }
    };

    this.ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      this.connected = false;
      if (this.onConnectionChange) this.onConnectionChange(false);
      this._scheduleReconnect();
    };

    this.ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };
  }

  _scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached');
      return;
    }
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.min(this.reconnectAttempts, 5);
    setTimeout(() => this.connect(), delay);
  }

  sendTelemetry(events) {
    const msg = { type: 'telemetry', events };
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    } else {
      this.queue.push(msg);
    }
  }

  sendLabel(label) {
    const msg = { type: 'label', label };
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    } else {
      this.queue.push(msg);
    }
  }

  disconnect() {
    this.maxReconnectAttempts = 0;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
  }
}
