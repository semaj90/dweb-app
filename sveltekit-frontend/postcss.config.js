export default {
  plugins: {
    '@unocss/postcss': {},
    'postcss-preset-env': {
      stage: 3,
      autoprefixer: {
        grid: true
      },
      features: {
        'custom-properties': true,
        // Removed deprecated 'color-mod-function'
        'color-function': false // Use modern color() function instead
      }
    }
  },
};