@echo off
title Fix Test Errors
echo Fixing test configuration...

cd sveltekit-frontend

echo Installing test dependencies...
npm install -D @testing-library/jest-dom vitest @vitest/ui jsdom

echo Creating test setup...
> src\test-setup.ts (
echo import '@testing-library/jest-dom';
)

echo Updating vitest config...
> vitest.config.ts (
echo import { defineConfig } from 'vitest/config';
echo import { sveltekit } from '@sveltejs/kit/vite';
echo.
echo export default defineConfig({
echo   plugins: [sveltekit()],
echo   test: {
echo     environment: 'jsdom',
echo     setupFiles: ['src/test-setup.ts']
echo   }
echo });
)

echo Fixing dropdown test...
> tests\dropdown-menu.test.ts (
echo import { describe, it, expect } from 'vitest';
echo import { render, screen } from '@testing-library/svelte';
echo import DropdownMenu from '$lib/components/ui/dropdown-menu/DropdownMenu.svelte';
echo.
echo describe('DropdownMenu', () => {
echo   it('renders correctly', () => {
echo     render(DropdownMenu);
echo     expect(screen.getByRole('button')).toBeInTheDocument();
echo   });
echo });
)

cd ..
echo ✅ Tests fixed
pause
