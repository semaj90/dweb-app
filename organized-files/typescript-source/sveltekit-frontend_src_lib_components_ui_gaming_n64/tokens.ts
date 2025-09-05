// N64 UI design tokens (colors, spacing, sizes, z-index)
export type ColorToken = string;
export type SizeToken = string | number;

export interface N64Tokens {
  colors: {
	primary: ColorToken;
	secondary: ColorToken;
	background: ColorToken;
	text: ColorToken;
  };
  spacing: {
	xs: string;
	sm: string;
	md: string;
	lg: string;
  };
  sizes: {
	button: SizeToken;
	icon: SizeToken;
  };
  zIndex: {
	dropdown: number;
	modal: number;
	tooltip: number;
  };
}

export const tokens: N64Tokens = {
  colors: {
	primary: '#0066cc',
	secondary: '#e5a400',
	background: '#0b1220',
	text: '#f5f7fa',
  },
  spacing: {
	xs: '4px',
	sm: '8px',
	md: '16px',
	lg: '24px',
  },
  sizes: {
	button: '40px',
	icon: 16,
  },
  zIndex: {
	dropdown: 1000,
	modal: 1100,
	tooltip: 1200,
  },
};

export default tokens;
