import type { Meta, StoryObj } from '@storybook/svelte';
import Button from './Button.svelte';

const meta = {
  title: 'UI/Button',
  component: Button,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'Primary UI button component with multiple variants for legal AI platform'
      }
    }
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'destructive', 'outline', 'secondary', 'ghost', 'link', 'legal', 'evidence', 'case']
    },
    size: {
      control: 'select',
      options: ['default', 'sm', 'lg', 'icon']
    },
    disabled: {
      control: 'boolean'
    },
    loading: {
      control: 'boolean',
      description: 'Show loading state with spinner'
    }
  }
} satisfies Meta<Button>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: 'Button'
  }
};

export const Legal: Story = {
  args: {
    variant: 'legal',
    children: 'Legal Action'
  },
  parameters: {
    docs: {
      description: {
        story: 'Legal-themed button variant for law-related actions'
      }
    }
  }
};

export const Evidence: Story = {
  args: {
    variant: 'evidence',
    children: 'Add Evidence'
  },
  parameters: {
    docs: {
      description: {
        story: 'Evidence-themed button for evidence management actions'
      }
    }
  }
};

export const Case: Story = {
  args: {
    variant: 'case',
    children: 'Create Case'
  },
  parameters: {
    docs: {
      description: {
        story: 'Case-themed button for case management actions'
      }
    }
  }
};

export const Destructive: Story = {
  args: {
    variant: 'destructive',
    children: 'Delete'
  },
  parameters: {
    docs: {
      description: {
        story: 'Destructive actions like deletion with warning styling'
      }
    }
  }
};

export const Outline: Story = {
  args: {
    variant: 'outline',
    children: 'Cancel'
  },
  parameters: {
    docs: {
      description: {
        story: 'Outline variant for secondary actions'
      }
    }
  }
};

export const Small: Story = {
  args: {
    size: 'sm',
    children: 'Small Button'
  }
};

export const Large: Story = {
  args: {
    size: 'lg',
    children: 'Large Button'
  }
};

export const Disabled: Story = {
  args: {
    disabled: true,
    children: 'Disabled Button'
  }
};

export const Loading: Story = {
  args: {
    loading: true,
    children: 'Loading...'
  },
  parameters: {
    docs: {
      description: {
        story: 'Button with loading state and spinner animation'
      }
    }
  }
};