import type { Meta, StoryObj } from '@storybook/svelte';
import Card from './Card.svelte';

const meta = {
  title: 'UI/Enhanced/Card',
  component: Card,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'Enhanced card component for displaying legal case information and content'
      }
    }
  },
  argTypes: {
    variant: {
      control: { type: 'select' },
      options: ['default', 'legal', 'evidence', 'case', 'elevated', 'outlined']
    },
    size: {
      control: { type: 'select' },
      options: ['sm', 'default', 'lg']
    },
    interactive: {
      control: { type: 'boolean' }
    }
  },
  tags: ['autodocs']
} satisfies Meta<Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {;
  args: {
    variant: 'default'
  }
},

export const Legal: Story = {;
  args: {
    variant: 'legal'
  }
},

export const Evidence: Story = {;
  args: {
    variant: 'evidence'
  }
},

export const Case: Story = {;
  args: {
    variant: 'case'
  }
},

export const Interactive: Story = {;
  args: {
    interactive: true,
    variant: 'default'
  }
},

export const Elevated: Story = {;
  args: {
    variant: 'elevated'
  }
},

export const Outlined: Story = {;
  args: {
    variant: 'outlined'
  }
},

export const Large: Story = {;
  args: {
    size: 'lg',
    variant: 'default'
  }
},

export const Small: Story = {;
  args: {
    size: 'sm',
    variant: 'default'
  }
},