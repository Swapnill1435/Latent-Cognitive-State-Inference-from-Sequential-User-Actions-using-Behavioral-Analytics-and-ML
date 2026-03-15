import React, { useState, useEffect, useRef, useCallback } from 'react';
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { BehavioralTracker } from './telemetry/tracker';
import { CognitiveWebSocket } from './telemetry/websocket';
import PuzzleTask from './tasks/PuzzleTask';
import DecisionTask from './tasks/DecisionTask';
import NavigationTask from './tasks/NavigationTask';
import Dashboard from './dashboard/Dashboard';
import AdaptiveUI from './adaptive/AdaptiveUI';

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
}

function AppContent() {
  const [sessionId] = useState(() => generateSessionId());
  const [connected, setConnected] = useState(false);
  const [cognitiveState, setCognitiveState] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const trackerRef = useRef(null);
  const wsRef = useRef(null);
  const location = useLocation();

  // Initialize WebSocket and tracker
  useEffect(() => {
    // Create session on backend
    fetch('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: 'demo_user' }),
    }).catch(() => {});

    // WebSocket
    const ws = new CognitiveWebSocket(sessionId, (msg) => {
      if (msg.type === 'cognitive_state') {
        setCognitiveState(msg.data);
        setPredictions(prev => [...prev.slice(-100), msg.data]);
      }
    });
    ws.onConnectionChange = setConnected;
    ws.connect();
    wsRef.current = ws;

    // Tracker
    const tracker = new BehavioralTracker((events) => {
      ws.sendTelemetry(events);
    });
    tracker.start();
    trackerRef.current = tracker;

    return () => {
      tracker.stop();
      ws.disconnect();
    };
  }, [sessionId]);

  // Track page navigation
  useEffect(() => {
    if (trackerRef.current) {
      trackerRef.current.trackCustom('page_navigation', {
        page: location.pathname,
      });
    }
  }, [location.pathname]);

  const currentState = cognitiveState?.predicted_state || null;
  const stateClass = currentState ? `state-${currentState}` : '';

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brain-icon">🧠</div>
          <h1>CogniSense</h1>
        </div>

        <nav>
          <ul className="sidebar-nav">
            <li>
              <NavLink to="/" end>
                <span className="nav-icon">📊</span> Dashboard
              </NavLink>
            </li>
            <li>
              <NavLink to="/task/puzzle">
                <span className="nav-icon">🧩</span> Puzzle Task
              </NavLink>
            </li>
            <li>
              <NavLink to="/task/decision">
                <span className="nav-icon">🤔</span> Decision Task
              </NavLink>
            </li>
            <li>
              <NavLink to="/task/navigation">
                <span className="nav-icon">🧭</span> Navigation Task
              </NavLink>
            </li>
          </ul>
        </nav>

        {/* Connection status */}
        <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
          <div className="connection-status">
            <span className={`connection-dot ${connected ? 'connected' : 'disconnected'}`}></span>
            {connected ? 'Connected' : 'Disconnected'}
          </div>

          {currentState && (
            <div style={{ marginTop: '0.5rem' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Cognitive State
              </div>
              <div className={`state-indicator ${stateClass}`}>
                <span className="state-dot"></span>
                {currentState}
              </div>
            </div>
          )}

          <div style={{ marginTop: '0.75rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Session: {sessionId.substring(0, 12)}...
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={
            <Dashboard
              sessionId={sessionId}
              cognitiveState={cognitiveState}
              predictions={predictions}
            />
          } />
          <Route path="/task/puzzle" element={
            <PuzzleTask tracker={trackerRef.current} />
          } />
          <Route path="/task/decision" element={
            <DecisionTask tracker={trackerRef.current} />
          } />
          <Route path="/task/navigation" element={
            <NavigationTask tracker={trackerRef.current} />
          } />
        </Routes>
      </main>

      {/* Adaptive UI Overlay */}
      <AdaptiveUI state={currentState} />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
