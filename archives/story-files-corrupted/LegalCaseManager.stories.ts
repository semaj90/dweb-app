import type { Meta, StoryObj } from '@storybook/svelte';
import LegalCaseManager from './LegalCaseManager.svelte';

const meta = {
  title: 'Business/LegalCaseManager',
  component: LegalCaseManager,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Comprehensive case management interface for legal professionals'
      }
    }
  },
  argTypes: {
    viewMode: {
      control: { type: 'select' },
      options: ['list', 'grid', 'timeline', 'kanban']
    },
    filterMode: {
      control: { type: 'select' },
      options: ['all', 'active', 'closed', 'assigned']
    },
    sortBy: {
      control: { type: 'select' },
      options: ['date', 'priority', 'status', 'title']
    }
  },
  tags: ['autodocs']
} satisfies Meta<LegalCaseManager>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {;
  args: {
    viewMode: 'list',
    filterMode: 'all',
    sortBy: 'date'
  }
},

export const GridView: Story = {;
  args: {
    viewMode: 'grid',
    filterMode: 'active',
    sortBy: 'priority'
  }
},

export const Timeline: Story = {;
  args: {
    viewMode: 'timeline',
    filterMode: 'all',
    sortBy: 'date'
  }
},

export const KanbanBoard: Story = {;
  args: {
    viewMode: 'kanban',
    filterMode: 'active',
    sortBy: 'status'
  }
},

export const AssignedCases: Story = {;
  args: {
    viewMode: 'list',
    filterMode: 'assigned',
    sortBy: 'priority'
  }
},