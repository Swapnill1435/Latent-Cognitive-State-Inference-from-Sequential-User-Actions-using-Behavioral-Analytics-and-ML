import React, { useState, useEffect, useRef } from 'react';

const STATE_COLORS = {
  confidence: '#22c55e',
  confused: '#ef4444',
  exploring: '#3b82f6',
  hesitating: '#f59e0b',
  overloaded: '#ec4899',
  fatigue: '#a78bfa',
};

function StateOverview({ cognitiveState }) {
  if (!cognitiveState) {
    return (
      <div className="card">
        <div className="card-header"><h3>🧠 Current State</h3></div>
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
          Interact with a task to start cognitive state inference...
        </p>
      </div>
    );
  }

  const { predicted_state, confidence, probabilities = {} } = cognitiveState;

  return (
    <div className="card">
      <div className="card-header">
        <h3>🧠 Current State</h3>
        <div className={`state-indicator state-${predicted_state}`}>
          <span className="state-dot"></span>
          {predicted_state}
        </div>
      </div>

      <div style={{ textAlign: 'center', margin: '1rem 0' }}>
        <div style={{ fontSize: 'var(--font-size-4xl)', fontWeight: 800, color: STATE_COLORS[predicted_state] }}>
          {(confidence * 100).toFixed(0)}%
        </div>
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          Confidence
        </div>
      </div>

      <div className="prob-bars">
        {Object.entries(probabilities).map(([state, prob]) => (
          <div key={state} className="prob-bar">
            <span className="label">{state}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${prob * 100}%`,
                  background: STATE_COLORS[state],
                }}
              />
            </div>
            <span className="value" style={{ color: STATE_COLORS[state] }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ModelComparison({ cognitiveState }) {
  if (!cognitiveState?.model_outputs) return null;
  const { hmm, lstm, transformer } = cognitiveState.model_outputs;
  const models = [
    { name: 'HMM', data: hmm, icon: '📊' },
    { name: 'LSTM', data: lstm, icon: '🔄' },
    { name: 'Transformer', data: transformer, icon: '⚡' },
  ];

  return (
    <div className="card">
      <div className="card-header"><h3>🔬 Model Comparison</h3></div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
        {models.map(({ name, data, icon }) => {
          const topState = data ? Object.entries(data).sort((a, b) => b[1] - a[1])[0] : null;
          return (
            <div key={name} style={{
              padding: '1rem', borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-glass)', border: '1px solid var(--border-subtle)',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{icon}</div>
              <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: '0.25rem' }}>{name}</div>
              {topState && (
                <>
                  <div style={{ color: STATE_COLORS[topState[0]], fontWeight: 700, textTransform: 'capitalize' }}>
                    {topState[0]}
                  </div>
                  <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>
                    {(topState[1] * 100).toFixed(1)}%
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StateTimeline({ predictions }) {
  if (predictions.length === 0) return null;
  const last50 = predictions.slice(-50);

  return (
    <div className="card full-width">
      <div className="card-header">
        <h3>📈 State Timeline</h3>
        <span className="badge">{predictions.length} predictions</span>
      </div>
      <div className="timeline">
        {last50.map((p, i) => (
          <div
            key={i}
            className="timeline-segment"
            style={{ background: STATE_COLORS[p.predicted_state] || '#666' }}
            title={`${p.predicted_state} (${(p.confidence * 100).toFixed(0)}%)`}
          />
        ))}
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem',
        fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)',
      }}>
        <span>Oldest</span>
        <span>Latest</span>
      </div>
    </div>
  );
}

function FeatureImportance({ sessionId }) {
  const [features, setFeatures] = useState([]);

  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        const res = await fetch(`/api/dashboard/sessions/${sessionId}/feature-importance`);
        if (res.ok) {
          const data = await res.json();
          setFeatures(data.features || []);
        }
      } catch (e) { /* silent */ }
    };

    const interval = setInterval(fetchFeatures, 3000);
    fetchFeatures();
    return () => clearInterval(interval);
  }, [sessionId]);

  const maxVal = Math.max(...features.map(f => Math.abs(f.value)), 1);

  return (
    <div className="card">
      <div className="card-header"><h3>📊 Behavioral Features</h3></div>
      {features.length === 0 ? (
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>
          No feature data yet...
        </p>
      ) : (
        <div style={{ maxHeight: '380px', overflowY: 'auto' }}>
          {features.map((f, i) => (
            <div key={i} className="feature-bar">
              <span className="fname">{f.feature.replace(/_/g, ' ')}</span>
              <div className="fbar">
                <div className="fbar-fill" style={{ width: `${(Math.abs(f.value) / maxVal) * 100}%` }} />
              </div>
              <span className="fval">{f.value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function InteractionHeatmap({ sessionId }) {
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);

  useEffect(() => {
    const fetchHeatmap = async () => {
      try {
        const res = await fetch(`/api/dashboard/sessions/${sessionId}/heatmap`);
        if (res.ok) {
          const data = await res.json();
          setPoints(data.heatmap || []);
        }
      } catch (e) { /* silent */ }
    };
    const interval = setInterval(fetchHeatmap, 5000);
    fetchHeatmap();
    return () => clearInterval(interval);
  }, [sessionId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = canvas.offsetHeight;

    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.fillRect(0, 0, w, h);

    const scaleX = w / 1920;
    const scaleY = h / 1080;

    points.forEach(p => {
      const x = p.x * scaleX;
      const y = p.y * scaleY;
      const r = p.type === 'click' ? 6 : 2;
      const alpha = p.type === 'click' ? 0.6 : 0.15;

      const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
      gradient.addColorStop(0, `rgba(99, 102, 241, ${alpha})`);
      gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, r * 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [points]);

  return (
    <div className="card">
      <div className="card-header">
        <h3>🗺️ Interaction Heatmap</h3>
        <span className="badge">{points.length} points</span>
      </div>
      <canvas ref={canvasRef} className="heatmap-canvas" />
    </div>
  );
}

function SessionStats({ sessionId }) {
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const res = await fetch(`/api/sessions/${sessionId}/analytics`);
        if (res.ok) {
          setAnalytics(await res.json());
        }
      } catch (e) { /* silent */ }
    };
    const interval = setInterval(fetchAnalytics, 5000);
    fetchAnalytics();
    return () => clearInterval(interval);
  }, [sessionId]);

  return (
    <div className="card full-width">
      <div className="card-header"><h3>📋 Session Analytics</h3></div>
      {!analytics ? (
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>Loading...</p>
      ) : (
        <div className="grid grid-4">
          <div className="stat-card">
            <div className="stat-value">{analytics.total_events}</div>
            <div className="stat-label">Total Events</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{analytics.total_predictions}</div>
            <div className="stat-label">Predictions</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{analytics.duration_seconds?.toFixed(0) || 0}s</div>
            <div className="stat-label">Duration</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{Object.keys(analytics.state_distribution || {}).length}</div>
            <div className="stat-label">States Detected</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Dashboard({ sessionId, cognitiveState, predictions }) {
  return (
    <div>
      <div className="page-header">
        <h2>📊 Analytics Dashboard</h2>
        <p>Real-time cognitive state inference and behavioral analytics</p>
      </div>

      <div className="dashboard-grid">
        <StateOverview cognitiveState={cognitiveState} />
        <ModelComparison cognitiveState={cognitiveState} />
        <StateTimeline predictions={predictions} />
        <FeatureImportance sessionId={sessionId} />
        <InteractionHeatmap sessionId={sessionId} />
        <SessionStats sessionId={sessionId} />
      </div>
    </div>
  );
}
