import React, { useState, useCallback, useEffect } from 'react';

const PAGES = {
  home: {
    title: '🏠 Information Hub',
    content: 'Welcome to the information hub. Find the answer to this question: "What year was the quantum computing breakthrough announced?" Navigate through the sections below to find it.',
    links: ['science', 'history', 'technology', 'about'],
  },
  science: {
    title: '🔬 Science Section',
    content: 'Latest breakthroughs in various scientific fields including biology, chemistry, and physics. The field of molecular biology has seen rapid advancement with CRISPR-Cas9 applications expanding beyond gene editing into diagnostics.',
    links: ['physics', 'biology', 'home'],
  },
  history: {
    title: '📜 History Section',
    content: 'Explore historical events and milestones across different eras. From ancient civilizations to modern history, each period offers unique insights into human progress.',
    links: ['ancient', 'modern', 'home'],
  },
  technology: {
    title: '💻 Technology Section',
    content: 'Cutting-edge technology news and analysis. Covering topics from artificial intelligence to space exploration and renewable energy systems.',
    links: ['ai', 'quantum', 'space', 'home'],
  },
  about: {
    title: 'ℹ️ About',
    content: 'This is an information navigation task. Your browsing patterns, backtracking, and exploration behavior are being analyzed.',
    links: ['home'],
  },
  physics: {
    title: '⚛️ Physics',
    content: 'Advances in theoretical and experimental physics. Recent work on quantum field theory has revealed new particles, and gravitational wave detectors continue to make groundbreaking observations.',
    links: ['quantum', 'science', 'home'],
  },
  biology: {
    title: '🧬 Biology',
    content: 'Breakthroughs in biological sciences. Synthetic biology is enabling the creation of entirely new organisms, while neuroscience maps the brain with unprecedented detail.',
    links: ['science', 'home'],
  },
  ancient: {
    title: '🏛️ Ancient History',
    content: 'Ancient civilizations and their contributions. The Mayans developed sophisticated astronomical observatories. Egyptian engineering achievements continue to fascinate researchers.',
    links: ['history', 'home'],
  },
  modern: {
    title: '🌍 Modern History',
    content: 'Key events from the 20th and 21st centuries. The digital revolution transformed communication, while space exploration opened new frontiers for humanity.',
    links: ['history', 'technology', 'home'],
  },
  ai: {
    title: '🤖 Artificial Intelligence',
    content: 'AI has made remarkable progress. Large language models can now engage in complex reasoning. Computer vision systems achieve superhuman accuracy in many domains.',
    links: ['technology', 'quantum', 'home'],
  },
  quantum: {
    title: '🔮 Quantum Computing',
    content: 'The quantum computing breakthrough was announced in 2029. A team of researchers achieved quantum advantage for practical optimization problems, solving in minutes what classical computers would need thousands of years to compute.',
    links: ['technology', 'physics', 'home'],
  },
  space: {
    title: '🚀 Space Exploration',
    content: 'Humanity continues to push the boundaries of space exploration. Mars missions are planned for the 2030s, and deep space telescopes reveal the earliest galaxies.',
    links: ['technology', 'physics', 'home'],
  },
};

export default function NavigationTask({ tracker }) {
  const [currentPage, setCurrentPage] = useState('home');
  const [visitHistory, setVisitHistory] = useState(['home']);
  const [startTime] = useState(Date.now());
  const [found, setFound] = useState(false);

  const page = PAGES[currentPage];
  const uniqueVisits = new Set(visitHistory).size;
  const backtracks = visitHistory.filter((p, i) => i > 0 && visitHistory.indexOf(p) < i).length;

  useEffect(() => {
    if (tracker) {
      tracker.trackCustom('nav_page_view', {
        page: currentPage,
        visit_count: visitHistory.filter(p => p === currentPage).length,
      });
    }
  }, [currentPage, tracker, visitHistory]);

  const navigate = useCallback((pageId) => {
    if (tracker) {
      tracker.trackCustom('nav_click', {
        from: currentPage,
        to: pageId,
        path: `/nav/${pageId}`,
      });
    }

    setVisitHistory(prev => [...prev, pageId]);
    setCurrentPage(pageId);

    if (pageId === 'quantum' && !found) {
      setFound(true);
      if (tracker) {
        tracker.trackCustom('nav_found_answer', {
          steps: visitHistory.length,
          time: Date.now() - startTime,
          backtracks,
        });
      }
    }
  }, [currentPage, tracker, visitHistory, found, startTime, backtracks]);

  const elapsed = Math.floor((Date.now() - startTime) / 1000);

  return (
    <div className="task-container">
      <div className="page-header">
        <h2>🧭 Navigation Task</h2>
        <p>Find the answer to the question by navigating through the information hub. Your navigation patterns reveal cognitive strategies.</p>
      </div>

      <div className="grid grid-4" style={{ marginBottom: '1.5rem' }}>
        <div className="card stat-card">
          <div className="stat-value">{visitHistory.length}</div>
          <div className="stat-label">Pages Visited</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{uniqueVisits}</div>
          <div className="stat-label">Unique Pages</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{backtracks}</div>
          <div className="stat-label">Backtracks</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{found ? '✅' : '🔍'}</div>
          <div className="stat-label">{found ? 'Found!' : 'Searching'}</div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3>{page.title}</h3>
          {found && currentPage === 'quantum' && (
            <span className="badge" style={{ background: 'rgba(34,197,94,0.15)', color: 'var(--state-focused)' }}>
              Answer Found!
            </span>
          )}
        </div>

        <div className="nav-task-page">
          <p style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.7, marginBottom: '1.5rem' }}>
            {page.content}
          </p>

          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Navigate to:
            </div>
            <div className="nav-task-links">
              {page.links.map(link => (
                <a key={link} onClick={() => navigate(link)}>
                  {PAGES[link]?.title?.split(' ').slice(1).join(' ') || link}
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Visit history trail */}
        <div style={{ marginTop: '1rem', fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>
          <span style={{ fontWeight: 600 }}>Path: </span>
          {visitHistory.slice(-8).map((p, i) => (
            <span key={i}>
              {i > 0 && ' → '}
              <span style={{ color: p === 'quantum' ? 'var(--state-focused)' : 'var(--text-secondary)' }}>
                {p}
              </span>
            </span>
          ))}
          {visitHistory.length > 8 && '...'}
        </div>
      </div>
    </div>
  );
}
