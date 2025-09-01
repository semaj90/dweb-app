import { describe, it, expect } from 'vitest';

interface SlashCommand { id: string; label: string; keywords?: string[] }

// Lightweight replicate of filtering logic used in the editor component
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
});
