import { describe, it, expect } from 'vitest';

interface SlashCommand { id: string; label: string; keywords?: string[]; action?: () => void }

// Replicate filtering logic from editor component
function filterCommands(commands: SlashCommand[], query: string): SlashCommand[] {
  const q = query.trim().toLowerCase();
  if (!q) return commands;
  return commands.filter(c =>
    c.label.toLowerCase().startsWith(q) ||
    (c.keywords?.some(k => k.toLowerCase().includes(q)))
  );
}

describe('slash menu filtering', () => {
  const commands: SlashCommand[] = [
    { id: 'h1', label: 'Heading 1', keywords: ['title','h1'] },
    { id: 'h2', label: 'Heading 2', keywords: ['subtitle','h2'] },
    { id: 'bullet', label: 'Bullet List', keywords: ['list','ul'] },
    { id: 'ai', label: 'AI Suggest', keywords: ['assistant','ai','review'] }
  ];

  it('returns all when query empty', () => {
    expect(filterCommands(commands, '').length).toBe(commands.length);
  });

  it('matches by label prefix', () => {
    const res = filterCommands(commands, 'hea');
    expect(res.map(r => r.id)).toEqual(['h1','h2']);
  });

  it('matches by keyword substring', () => {
    const res = filterCommands(commands, 'ass');
    expect(res.map(r => r.id)).toEqual(['ai']);
  });

  it('returns empty when no match', () => {
    const res = filterCommands(commands, 'zzz');
    expect(res.length).toBe(0);
  });

  it('keyboard wrap-around logic (simulated)', () => {
    // simulate index cycling
    const len = commands.length;
    let index = 0;
    function down(){ index = (index + 1) % len; }
    function up(){ index = (index - 1 + len) % len; }
  down(); down(); down(); down(); // 4 downs -> index should be 0
  expect(index).toBe(0);
  down(); // wrap cycle again -> index 1
  expect(index).toBe(1);
  up(); // move back from 1 -> 0 (no wrap yet)
  expect(index).toBe(0);
  up(); // wrap backwards from 0 -> len-1
  expect(index).toBe(len - 1);
  });

  it('executes correct command (simulation)', () => {
    const executed: string[] = [];
    const extended = commands.map(c => ({ ...c, action: () => executed.push(c.id) }));
    // pick third filtered after query 'h'
    const filtered = filterCommands(extended as any, 'h');
    expect(filtered.map(f=>f.id)).toEqual(['h1','h2']);
    filtered[1].action();
    expect(executed).toEqual(['h2']);
  });
});
