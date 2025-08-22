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
        'color-mod-function': { unresolved: 'warn' }
      }
    }
  },
};