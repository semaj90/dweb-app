/**
 * jscodeshift codemod: transform-glob-and-safety
 *
 * - Rewrites `import glob from 'glob'` -> `import { glob } from 'glob'`
 * - Converts simple `const x = require('mod')` and `const { a } = require('mod')` to ESM imports
 * - Normalizes `throw { ... }` -> `throw new Error(JSON.stringify({...}))` (or uses `.message` if present)
 * - Rewrites `error(status, { ... })` -> `error(status, new Error(...))` for SvelteKit usage
 *
 * Usage (install jscodeshift globally or use npx):
 *   npx jscodeshift -t scripts/codemods/transform-glob-and-safety.js <paths>
 * Example:
 *   npx jscodeshift -t scripts/codemods/transform-glob-and-safety.js sveltekit-frontend/src
 */

module.exports = function transformer(file, api) {
  const j = api.jscodeshift;
  const root = j(file.source);

  // 1) transform default glob import -> named import { glob }
  root.find(j.ImportDeclaration, { source: { value: 'glob' } })
    .forEach(p => {
      const decl = p.node;
      const specs = decl.specifiers || [];
      if (specs.length === 1 && specs[0].type === 'ImportDefaultSpecifier') {
        decl.specifiers = [j.importSpecifier(j.identifier('glob'))];
      }
    });

  // 2) transform common require() patterns to ESM imports
  root.find(j.VariableDeclaration)
    .forEach(path => {
      const decl = path.node;
      const toRemove = [];
      const imports = [];

      decl.declarations.forEach(d => {
        if (d.init && d.init.type === 'CallExpression' && d.init.callee.name === 'require' && d.init.arguments.length === 1 && d.init.arguments[0].type === 'Literal') {
          const mod = d.init.arguments[0].value;
          if (d.id.type === 'Identifier') {
            // const fs = require('fs') -> import * as fs from 'fs';
            imports.push(j.importDeclaration([j.importNamespaceSpecifier(j.identifier(d.id.name))], j.literal(mod)));
            toRemove.push(d);
          } else if (d.id.type === 'ObjectPattern') {
            // const { readFile } = require('fs') -> import { readFile } from 'fs';
            const specifiers = d.id.properties.map(p => {
              if (p.type === 'Property') {
                const local = p.value && p.value.name ? p.value.name : (p.key && p.key.name ? p.key.name : null);
                const imported = p.key && (p.key.name || p.key.value);
                if (!imported) return null;
                if (local && local !== imported) return j.importSpecifier(j.identifier(imported), j.identifier(local));
                return j.importSpecifier(j.identifier(imported));
              }
              return null;
            }).filter(Boolean);
            if (specifiers.length) imports.push(j.importDeclaration(specifiers, j.literal(mod)));
            toRemove.push(d);
          }
        }
      });

      if (imports.length) {
        path.insertBefore(imports);
        if (toRemove.length === decl.declarations.length) {
          j(path).remove();
        } else {
          decl.declarations = decl.declarations.filter(d => !toRemove.includes(d));
        }
      }
    });

  // 3) normalize throw object literals -> throw new Error(message || JSON.stringify(obj))
  root.find(j.ThrowStatement)
    .forEach(p => {
      const arg = p.node.argument;
      if (arg && arg.type === 'ObjectExpression') {
        const msgProp = arg.properties.find(pr => pr.type === 'Property' && ((pr.key.type === 'Identifier' && pr.key.name === 'message') || (pr.key.type === 'Literal' && pr.key.value === 'message')));
        let newArg;
        if (msgProp) {
          newArg = j.newExpression(j.identifier('Error'), [msgProp.value]);
        } else {
          newArg = j.newExpression(j.identifier('Error'), [j.callExpression(j.memberExpression(j.identifier('JSON'), j.identifier('stringify')), [arg])]);
        }
        p.replace(j.throwStatement(newArg));
      }
    });

  // 4) SvelteKit error(status, { ... }) -> error(status, new Error(...))
  root.find(j.CallExpression, { callee: { type: 'Identifier', name: 'error' } })
    .forEach(p => {
      const args = p.node.arguments || [];
      if (args.length >= 2 && args[1].type === 'ObjectExpression') {
        const obj = args[1];
        const msgProp = obj.properties.find(pr => pr.type === 'Property' && ((pr.key.type === 'Identifier' && pr.key.name === 'message') || (pr.key.type === 'Literal' && pr.key.value === 'message')));
        let newExpr;
        if (msgProp) {
          newExpr = j.newExpression(j.identifier('Error'), [msgProp.value]);
        } else {
          newExpr = j.newExpression(j.identifier('Error'), [j.callExpression(j.memberExpression(j.identifier('JSON'), j.identifier('stringify')), [obj])]);
        }
        p.node.arguments[1] = newExpr;
      }
    });

  return root.toSource({ quote: 'single' });
};
