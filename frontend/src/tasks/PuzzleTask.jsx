import React, { useState, useCallback, useEffect } from 'react';

function createPuzzle() {
  // Create a solvable 15-puzzle
  const tiles = Array.from({ length: 15 }, (_, i) => i + 1);
  tiles.push(0); // empty
  // Shuffle (simple Fisher-Yates)
  for (let i = tiles.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [tiles[i], tiles[j]] = [tiles[j], tiles[i]];
  }
  return tiles;
}

function isSolved(tiles) {
  for (let i = 0; i < 15; i++) {
    if (tiles[i] !== i + 1) return false;
  }
  return tiles[15] === 0;
}

export default function PuzzleTask({ tracker }) {
  const [tiles, setTiles] = useState(() => createPuzzle());
  const [moves, setMoves] = useState(0);
  const [startTime] = useState(Date.now());
  const [solved, setSolved] = useState(false);

  const emptyIndex = tiles.indexOf(0);

  const canMove = useCallback((index) => {
    const row = Math.floor(index / 4);
    const col = index % 4;
    const emptyRow = Math.floor(emptyIndex / 4);
    const emptyCol = emptyIndex % 4;
    return (
      (Math.abs(row - emptyRow) === 1 && col === emptyCol) ||
      (Math.abs(col - emptyCol) === 1 && row === emptyRow)
    );
  }, [emptyIndex]);

  const handleTileClick = useCallback((index) => {
    if (solved || !canMove(index)) return;

    if (tracker) {
      tracker.trackCustom('puzzle_move', {
        tile: tiles[index],
        from: index,
        to: emptyIndex,
        x: (index % 4) * 80 + 40,
        y: Math.floor(index / 4) * 80 + 40,
      });
    }

    const newTiles = [...tiles];
    [newTiles[index], newTiles[emptyIndex]] = [newTiles[emptyIndex], newTiles[index]];
    setTiles(newTiles);
    setMoves(m => m + 1);

    if (isSolved(newTiles)) {
      setSolved(true);
      if (tracker) {
        tracker.trackCustom('puzzle_solved', {
          moves: moves + 1,
          time: Date.now() - startTime,
        });
      }
    }
  }, [tiles, emptyIndex, solved, canMove, tracker, moves, startTime]);

  const resetPuzzle = () => {
    setTiles(createPuzzle());
    setMoves(0);
    setSolved(false);
    if (tracker) tracker.trackCustom('puzzle_reset', {});
  };

  const elapsed = Math.floor((Date.now() - startTime) / 1000);

  return (
    <div className="task-container">
      <div className="page-header">
        <h2>🧩 Sliding Puzzle</h2>
        <p>Arrange the tiles in order from 1—15. Your interaction patterns are being analyzed in real-time.</p>
      </div>

      <div className="grid grid-3" style={{ marginBottom: '1.5rem' }}>
        <div className="card stat-card">
          <div className="stat-value">{moves}</div>
          <div className="stat-label">Moves</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{elapsed}s</div>
          <div className="stat-label">Time</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">{solved ? '✅' : '🔄'}</div>
          <div className="stat-label">{solved ? 'Solved!' : 'In Progress'}</div>
        </div>
      </div>

      <div className="card">
        <div className="task-area">
          <div className="puzzle-grid">
            {tiles.map((tile, index) => (
              <div
                key={index}
                className={`puzzle-tile ${tile === 0 ? 'empty' : 'filled'}`}
                onClick={() => handleTileClick(index)}
                style={tile === 0 ? {} : {
                  background: `linear-gradient(135deg, hsl(${tile * 24}, 70%, 50%), hsl(${tile * 24 + 30}, 70%, 40%))`,
                }}
              >
                {tile !== 0 ? tile : ''}
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.5rem' }}>
          <button className="btn btn-secondary" onClick={resetPuzzle}>
            🔄 New Puzzle
          </button>
        </div>
      </div>
    </div>
  );
}
