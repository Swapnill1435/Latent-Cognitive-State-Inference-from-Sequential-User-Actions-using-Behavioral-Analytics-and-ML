import React, { useState, useEffect, useCallback } from 'react';

const SCENARIOS = [
  {
    id: 1,
    difficulty: 2,
    question: "You're planning a weekend trip. Which destination would you choose?",
    options: [
      { id: 'a', text: 'A cozy mountain cabin — quiet, scenic trails, perfect for reflection' },
      { id: 'b', text: 'A bustling city — museums, restaurants, vibrant nightlife' },
      { id: 'c', text: 'A beach resort — relaxation, ocean views, water sports' },
      { id: 'd', text: 'A countryside farm stay — fresh air, animals, authentic food' },
    ],
  },
  {
    id: 2,
    difficulty: 3,
    question: "A project deadline is approaching. You have limited time. How do you prioritize?",
    options: [
      { id: 'a', text: 'Focus on the core feature and deliver a minimal but polished version' },
      { id: 'b', text: 'Try to complete everything but accept some rough edges' },
      { id: 'c', text: 'Request a deadline extension to deliver quality work' },
      { id: 'd', text: 'Delegate parts to teammates and coordinate the integration' },
    ],
  },
  {
    id: 3,
    difficulty: 4,
    question: "You receive two job offers simultaneously. How do you decide?",
    options: [
      { id: 'a', text: 'Take the one with higher salary, even if the work is less exciting' },
      { id: 'b', text: 'Choose the role aligned with your passion, even at lower pay' },
      { id: 'c', text: 'Negotiate with both companies before making a final decision' },
      { id: 'd', text: 'Ask trusted mentors for advice and decide based on long-term growth' },
    ],
  },
  {
    id: 4,
    difficulty: 5,
    question: "An ethical dilemma: Your team discovers a data breach but reporting it will delay the product launch by months. What do you do?",
    options: [
      { id: 'a', text: 'Report immediately — user safety is non-negotiable' },
      { id: 'b', text: 'Fix it quietly and report after the launch' },
      { id: 'c', text: 'Assess the severity first, then decide on the appropriate timeline' },
      { id: 'd', text: 'Escalate to management and let them make the call' },
    ],
  },
];

export default function DecisionTask({ tracker }) {
  const [currentScenario, setCurrentScenario] = useState(0);
  const [selected, setSelected] = useState(null);
  const [answers, setAnswers] = useState([]);
  const [viewTime, setViewTime] = useState(Date.now());
  const [changeCount, setChangeCount] = useState(0);

  const scenario = SCENARIOS[currentScenario];

  useEffect(() => {
    setViewTime(Date.now());
    if (tracker) {
      tracker.trackCustom('decision_view', {
        scenario_id: scenario.id,
        difficulty: scenario.difficulty,
      });
    }
  }, [currentScenario, tracker, scenario]);

  const handleSelect = useCallback((optionId) => {
    const hesitation = Date.now() - viewTime;

    if (selected !== null && selected !== optionId) {
      setChangeCount(c => c + 1);
      if (tracker) {
        tracker.trackCustom('answer_change', {
          scenario_id: scenario.id,
          from: selected,
          to: optionId,
          hesitation,
        });
      }
    } else if (selected === null) {
      if (tracker) {
        tracker.trackCustom('answer_select', {
          scenario_id: scenario.id,
          option: optionId,
          hesitation,
        });
      }
    }

    setSelected(optionId);
  }, [selected, viewTime, tracker, scenario]);

  const handleNext = () => {
    setAnswers(prev => [...prev, {
      scenario_id: scenario.id,
      selected,
      changes: changeCount,
      time: Date.now() - viewTime,
    }]);

    if (tracker) {
      tracker.trackCustom('decision', {
        scenario_id: scenario.id,
        selected,
        changes: changeCount,
        hesitation: Date.now() - viewTime,
      });
    }

    if (currentScenario < SCENARIOS.length - 1) {
      setCurrentScenario(c => c + 1);
      setSelected(null);
      setChangeCount(0);
    }
  };

  const isLastScenario = currentScenario >= SCENARIOS.length - 1;
  const allDone = isLastScenario && selected !== null && answers.length >= SCENARIOS.length;

  return (
    <div className="task-container">
      <div className="page-header">
        <h2>🤔 Decision Making</h2>
        <p>Read each scenario and choose the option that best matches your approach. Take your time — hesitation patterns are captured.</p>
      </div>

      <div className="grid grid-4" style={{ marginBottom: '1.5rem' }}>
        <div className="card stat-card">
          <div className="stat-value">{currentScenario + 1}/{SCENARIOS.length}</div>
          <div className="stat-label">Scenario</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">⭐{scenario.difficulty}</div>
          <div className="stat-label">Difficulty</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{changeCount}</div>
          <div className="stat-label">Changes</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{Math.floor((Date.now() - viewTime) / 1000)}s</div>
          <div className="stat-label">Thinking</div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>Scenario {currentScenario + 1}</h3>
          <span className="badge">Difficulty {scenario.difficulty}/5</span>
        </div>

        <p style={{ fontSize: 'var(--font-size-lg)', marginBottom: '1.5rem', lineHeight: 1.6 }}>
          {scenario.question}
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {scenario.options.map((opt) => (
            <div
              key={opt.id}
              className={`decision-card ${selected === opt.id ? 'selected' : ''}`}
              onClick={() => handleSelect(opt.id)}
            >
              <span style={{
                display: 'inline-flex', width: 28, height: 28, borderRadius: '50%',
                background: selected === opt.id ? 'var(--accent-gradient)' : 'var(--bg-hover)',
                alignItems: 'center', justifyContent: 'center', marginRight: '0.75rem',
                fontSize: 'var(--font-size-sm)', fontWeight: 700, color: 'white',
              }}>
                {opt.id.toUpperCase()}
              </span>
              {opt.text}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.5rem', gap: '0.75rem' }}>
          {!isLastScenario && (
            <button
              className="btn btn-primary"
              disabled={!selected}
              onClick={handleNext}
              style={{ opacity: selected ? 1 : 0.5 }}
            >
              Next Scenario →
            </button>
          )}
          {isLastScenario && selected && (
            <button className="btn btn-primary" onClick={handleNext}>
              ✅ Complete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
