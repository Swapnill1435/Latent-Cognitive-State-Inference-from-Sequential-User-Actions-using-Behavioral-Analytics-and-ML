/**
 * High-resolution behavioral telemetry tracker.
 * Captures mouse moves (throttled), clicks, scrolls, keystrokes, hovers, and navigation.
 * Uses performance.now() for millisecond-precision timestamps.
 */

const THROTTLE_MS = 16; // ~60Hz

export class BehavioralTracker {
  constructor(onEvents) {
    this.onEvents = onEvents; // callback(events[])
    this.buffer = [];
    this.flushInterval = null;
    this.lastMoveTime = 0;
    this.tracking = false;
    this.startTime = performance.now();

    this._onMouseMove = this._onMouseMove.bind(this);
    this._onClick = this._onClick.bind(this);
    this._onScroll = this._onScroll.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onMouseOver = this._onMouseOver.bind(this);
  }

  _timestamp() {
    return performance.now() - this.startTime;
  }

  start() {
    if (this.tracking) return;
    this.tracking = true;
    this.startTime = performance.now();

    document.addEventListener('mousemove', this._onMouseMove, { passive: true });
    document.addEventListener('click', this._onClick, { passive: true });
    document.addEventListener('scroll', this._onScroll, { passive: true });
    document.addEventListener('keydown', this._onKeyDown, { passive: true });
    document.addEventListener('mouseover', this._onMouseOver, { passive: true });

    // Flush buffer every 500ms
    this.flushInterval = setInterval(() => this._flush(), 500);
  }

  stop() {
    this.tracking = false;
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('click', this._onClick);
    document.removeEventListener('scroll', this._onScroll);
    document.removeEventListener('keydown', this._onKeyDown);
    document.removeEventListener('mouseover', this._onMouseOver);

    if (this.flushInterval) {
      clearInterval(this.flushInterval);
      this.flushInterval = null;
    }
    this._flush();
  }

  _emit(event) {
    this.buffer.push(event);
    if (this.buffer.length >= 50) {
      this._flush();
    }
  }

  _flush() {
    if (this.buffer.length > 0 && this.onEvents) {
      this.onEvents([...this.buffer]);
      this.buffer = [];
    }
  }

  _onMouseMove(e) {
    const now = this._timestamp();
    if (now - this.lastMoveTime < THROTTLE_MS) return;
    this.lastMoveTime = now;

    this._emit({
      type: 'mousemove',
      timestamp: now,
      x: e.clientX,
      y: e.clientY,
      path: window.location.pathname,
    });
  }

  _onClick(e) {
    this._emit({
      type: 'click',
      timestamp: this._timestamp(),
      x: e.clientX,
      y: e.clientY,
      target: e.target?.tagName || '',
      path: window.location.pathname,
    });
  }

  _onScroll() {
    this._emit({
      type: 'scroll',
      timestamp: this._timestamp(),
      scrollY: window.scrollY,
      scrollX: window.scrollX,
      path: window.location.pathname,
    });
  }

  _onKeyDown(e) {
    this._emit({
      type: 'keystroke',
      timestamp: this._timestamp(),
      key: e.key.length === 1 ? 'char' : e.key, // anonymize actual keys
      path: window.location.pathname,
    });
  }

  _onMouseOver(e) {
    if (e.target?.tagName === 'BUTTON' || e.target?.tagName === 'A') {
      this._emit({
        type: 'hover',
        timestamp: this._timestamp(),
        x: e.clientX,
        y: e.clientY,
        target: e.target?.tagName || '',
        path: window.location.pathname,
      });
    }
  }

  // Custom event injection (e.g., answer change, decision view)
  trackCustom(eventType, data = {}) {
    this._emit({
      type: eventType,
      timestamp: this._timestamp(),
      ...data,
      path: window.location.pathname,
    });
  }
}
