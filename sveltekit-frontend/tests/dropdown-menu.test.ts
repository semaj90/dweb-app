import { render, screen } from '@testing-library/svelte';
import { expect, test, describe } from 'vitest';
import DropdownMenu from '$lib/components/ui/DropdownMenu.svelte';

describe('DropdownMenu Component', () => {
  const mockItems = [
    { label: 'Option 1', value: 'option1' },
    { label: 'Option 2', value: 'option2' },
    { separator: true, label: '', value: '' },
    { label: 'Option 3', value: 'option3', disabled: true }
  ];

  test('renders trigger with correct text', () => {
    render(DropdownMenu, {
      props: {
        items: mockItems,
        trigger: 'Test Menu'
      }
    });

    expect(screen.getByText('Test Menu')).toBeInTheDocument();
  });

  test('shows dropdown items when opened', async () => {
    const { component } = render(DropdownMenu, {
      props: {
        items: mockItems,
        trigger: 'Test Menu'
      }
    });

    // Trigger should be visible
    const trigger = screen.getByTestId('dropdown-trigger');
    expect(trigger).toBeInTheDocument();

    // Click trigger to open
    await trigger.click();

    // Items should be visible
    expect(screen.getByText('Option 1')).toBeInTheDocument();
    expect(screen.getByText('Option 2')).toBeInTheDocument();
    expect(screen.getByText('Option 3')).toBeInTheDocument();
  });

  test('calls onSelect when item is clicked', async () => {
    let selectedValue = '';
ECHO is on.
    render(DropdownMenu, {
      props: {
        items: mockItems,
        trigger: 'Test Menu',
        onSelect: (value) => { selectedValue = value; }
      }
    });

    const trigger = screen.getByTestId('dropdown-trigger');
    await trigger.click();

    const option1 = screen.getByText('Option 1');
    await option1.click();

    expect(selectedValue).toBe('option1');
  });

  test('handles disabled items correctly', async () => {
    render(DropdownMenu, {
      props: {
        items: mockItems,
        trigger: 'Test Menu'
      }
    });

    const trigger = screen.getByTestId('dropdown-trigger');
    await trigger.click();

    const disabledOption = screen.getByText('Option 3');
    expect(disabledOption).toHaveAttribute('data-disabled');
  });
});
