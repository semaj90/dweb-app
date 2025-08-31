/**
 * jscodeshift transform: convert default `import glob from 'glob'` to `import { glob } from 'glob'`
 * Safer than regex — uses AST to rewrite only default imports from the "glob" package.
 *
 * Usage:
 * npx jscodeshift -t scripts/codemods/transform-glob-import.js src --extensions=ts,js,tsx --parser=tsx
 */

module.exports = function transformer(file, api) {
  const j = api.jscodeshift;
  const root = j(file.source);

  root.find(j.ImportDeclaration, { source: { value: 'glob' } }).forEach((path) => {
    const specifiers = path.node.specifiers || [];

    // If there's exactly one default import (import glob from 'glob')
    if (
      specifiers.length === 1 &&
      specifiers[0].type === 'ImportDefaultSpecifier'
    ) {
      // Replace default import with named import { glob }
      path.node.specifiers = [j.importSpecifier(j.identifier('glob'))];
    }
  });

  return root.toSource({ quote: 'single' });
};
