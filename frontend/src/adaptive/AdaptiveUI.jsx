import React, { useState, useEffect } from 'react';

/**
 * Adaptive UI wrapper that responds to predicted cognitive states:
 * - Confusion → contextual hints
 * - Cognitive overload → reduce information, suggest simplification
 * - Focused → minimal intervention
 */

const HINTS = {
  confused: {
    icon: '💡',
    title: 'Need a hint?',
    messages: [
      'Try breaking down the problem into smaller steps.',
      'Look at the patterns carefully — there may be a simpler path.',
      'Consider going back and re-reading the instructions.',
      'Take a deep breath and approach from a different angle.',
    ],
  },
  overloaded: {
    icon: '🧘',
    title: 'Take it easy',
    messages: [
      'You seem to be handling a lot. Consider focusing on one thing at a time.',
      'The interface has been simplified to reduce cognitive load.',
      'Try pausing for a moment before continuing.',
      'Focus on the most important element on the screen.',
    ],
  },
  hesitating: {
    icon: '⏰',
    title: 'Take your time',
    messages: [
      "It's okay to take time with decisions. There's no wrong answer.",
      'Trust your first instinct — it is often right.',
      'Consider what matters most to you, then choose.',
    ],
  },
  fatigue: {
    icon: '😴',
    title: 'Feeling tired?',
    messages: [
      'Your response patterns suggest fatigue. Consider taking a short break.',
      'Rest improves performance — a 5-minute break can help.',
      'Try standing up, stretching, or getting some water.',
      'You\'ve been working for a while. Your brain needs rest to perform well.',
    ],
  },
};

export default function AdaptiveUI({ state }) {
  const [visible, setVisible] = useState(false);
  const [dismissed, setDismissed] = useState(new Set());
  const [messageIndex, setMessageIndex] = useState(0);
  const [cooldown, setCooldown] = useState(false);

  useEffect(() => {
    if (!state || state === 'confidence' || state === 'exploring') {
      setVisible(false);
      return;
    }

    if (dismissed.has(state) || cooldown) {
      return;
    }

    // Show hint after a short delay
    const timer = setTimeout(() => {
      setVisible(true);
      setMessageIndex(Math.floor(Math.random() * (HINTS[state]?.messages?.length || 1)));
    }, 2000);

    return () => clearTimeout(timer);
  }, [state, dismissed, cooldown]);

  const handleDismiss = () => {
    setVisible(false);
    setDismissed(prev => new Set([...prev, state]));
    setCooldown(true);
    // Reset cooldown after 30 seconds
    setTimeout(() => setCooldown(false), 30000);
    // Reset dismissals after 60 seconds
    setTimeout(() => setDismissed(new Set()), 60000);
  };

  if (!visible || !state) return null;

  const hint = HINTS[state];
  if (!hint) return null;

  return (
    <div className={`adaptive-hint ${state === 'confused' ? 'confusion' : state === 'overloaded' ? 'overload' : ''}`}>
      <button className="hint-close" onClick={handleDismiss}>✕</button>
      <div className="hint-title">
        <span>{hint.icon}</span>
        {hint.title}
      </div>
      <div className="hint-text">
        {hint.messages[messageIndex % hint.messages.length]}
      </div>
    </div>
  );
}
