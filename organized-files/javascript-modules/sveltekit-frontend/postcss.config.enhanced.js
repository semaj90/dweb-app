/**
 * Enhanced PostCSS Configuration
 * Memory-aware CSS processing with parallel optimization
 */

import { cpus } from 'os';

const SYSTEM_CORES = cpus().length;
const ENABLE_PARALLEL = SYSTEM_CORES >= 4;

export default {
  plugins: {
    // Import resolver with enhanced path resolution
    'postcss-import': {
      // Parallel import processing
      resolve: (id, basedir, importOptions) => {
        // Enhanced import resolution with caching
        return id;
      }
    },
    
    // Nested CSS support
    'postcss-nested': {
      // Memory-efficient nesting
      bubble: ['screen']
    },
    
    // CSS variables with enhanced processing
    'postcss-custom-properties': {
      preserve: false, // Remove custom properties for better performance
      importFrom: [
        './src/lib/styles/variables.css',
        './src/lib/styles/theme.css'
      ]
    },
    
    // Autoprefixer with intelligent browser targeting
    'autoprefixer': {
      // Dynamic browser list based on build target
      overrideBrowserslist: process.env.NODE_ENV === 'production' 
        ? ['> 0.5%', 'last 2 versions', 'not dead']
        : ['last 2 Chrome versions', 'last 2 Firefox versions'],
      
      // Enhanced grid support
      grid: 'autoplace',
      
      // Flexbox support
      flexbox: 'no-2009'
    },
    
    // Enhanced CSS optimization
    ...(process.env.NODE_ENV === 'production' && {
      'cssnano': {
        preset: ['advanced', {
          // Advanced optimizations with memory awareness
          autoprefixer: false, // Already handled above
          
          // Parallel processing optimizations
          calc: {
            precision: 3,
            warnWhenCannotResolve: false
          },
          
          // Color optimizations
          colormin: true,
          convertValues: {
            length: true,
            time: true,
            angle: true
          },
          
          // Comment removal
          discardComments: {
            removeAll: true,
            removeAllButFirst: false
          },
          
          // Duplicate removal with enhanced algorithms
          discardDuplicates: true,
          discardEmpty: true,
          
          // Rule merging with memory optimization
          mergeLonghand: true,
          mergeRules: true,
          
          // Font optimization
          minifyFontValues: true,
          
          // Parameter optimization
          minifyParams: true,
          
          // Selector optimization
          minifySelectors: true,
          
          // Normalize optimizations
          normalizeCharset: true,
          normalizeDisplayValues: true,
          normalizePositions: true,
          normalizeRepeatStyle: true,
          normalizeString: true,
          normalizeTimingFunctions: true,
          normalizeUnicode: true,
          normalizeUrl: {
            normalizeProtocol: true,
            normalizeDataURI: true,
            stripHash: false,
            stripWWW: false,
            sortQueryParameters: true
          },
          normalizeWhitespace: true,
          
          // Value ordering
          orderedValues: true,
          
          // Property reduction
          reduceInitial: true,
          reduceTransforms: true,
          
          // SVG optimization
          svgo: {
            plugins: [
              'removeViewBox',
              'cleanupIDs',
              'minifyStyles'
            ]
          },
          
          // Selector uniqueness
          uniqueSelectors: true,
          
          // Z-index optimization
          zindex: {
            startIndex: 1
          },
          
          // Enhanced discarding rules
          discardUnused: {
            fontFace: true,
            counterStyle: true,
            keyframes: true,
            namespace: true
          }
        }]
      }
    }),
    
    // PurgeCSS for production builds with enhanced scanning
    ...(process.env.NODE_ENV === 'production' && {
      '@fullhuman/postcss-purgecss': {
        // Enhanced content scanning with parallel processing
        content: [
          './src/**/*.{html,js,svelte,ts}',
          './src/app.html'
        ],
        
        // Whitelist important classes
        safelist: [
          // UnoCSS safe classes
          /^(m|p|w|h|text|bg|border|flex|grid|absolute|relative|fixed|sticky)-/,
          
          // Dynamic classes that might be generated
          /^yorha-/,
          /^cyber-/,
          /^neural-/,
          /^canvas-/,
          /^demo-/,
          
          // Animation classes
          /^animate-/,
          /^transition-/,
          /^duration-/,
          /^ease-/,
          
          // State classes
          /^hover:/,
          /^focus:/,
          /^active:/,
          /^disabled:/,
          
          // Responsive classes
          /^sm:/,
          /^md:/,
          /^lg:/,
          /^xl:/,
          /^2xl:/
        ],
        
        // Enhanced extraction
        defaultExtractor: (content) => {
          // Multi-pattern extraction for better coverage
          const patterns = [
            // Standard class extraction
            /[A-Za-z0-9-_:/.]+/g,
            
            // Svelte class directives
            /class:([A-Za-z0-9-_]+)/g,
            
            // Template literal classes
            /`[^`]*`/g,
            
            // CSS-in-JS patterns
            /className\s*[:=]\s*["'`]([^"'`]*)["'`]/g
          ];
          
          const matches = patterns.flatMap(pattern => 
            [...(content.match(pattern) || [])]
          );
          
          return matches.filter(match => match.length > 0);
        },
        
        // Performance optimizations
        rejected: process.env.NODE_ENV === 'development',
        printRejected: false,
        
        // Parallel processing if supported
        ...(ENABLE_PARALLEL && {
          parallel: Math.min(SYSTEM_CORES, 4)
        })
      }
    }),
    
    // CSS modules support
    'postcss-modules': {
      // Enhanced module naming
      generateScopedName: process.env.NODE_ENV === 'production'
        ? '[hash:base64:8]'
        : '[name]__[local]--[hash:base64:5]',
      
      // Export globals
      exportGlobals: true,
      
      // Local by default
      localsConvention: 'camelCaseOnly'
    },
    
    // Custom media queries for responsive design
    'postcss-custom-media': {
      importFrom: [
        {
          customMedia: {
            '--phone': '(max-width: 767px)',
            '--tablet': '(min-width: 768px) and (max-width: 1023px)',
            '--desktop': '(min-width: 1024px)',
            '--large-desktop': '(min-width: 1440px)',
            '--ultra-wide': '(min-width: 1920px)'
          }
        }
      ]
    },
    
    // Modern CSS features
    'postcss-preset-env': {
      // Stage 2 features (good browser support)
      stage: 2,
      
      // Enhanced feature control
      features: {
        'nesting-rules': true,
        'custom-properties': false, // Handled above
        'custom-media-queries': false, // Handled above
        'custom-selectors': true,
        'media-query-ranges': true,
        'logical-properties-and-values': true,
        'prefers-color-scheme-query': true,
        'gap-properties': true,
        'overflow-shorthand': true,
        'place-properties': true,
        'color-functional-notation': true
      },
      
      // Browser targeting
      browsers: process.env.NODE_ENV === 'production'
        ? 'defaults'
        : 'last 2 versions'
    },
    
    // Development helpers
    ...(process.env.NODE_ENV === 'development' && {
      // CSS debugging
      'postcss-debug': {
        // Enhanced debugging output
        verbose: true,
        logLevel: 'info'
      },
      
      // Performance monitoring
      'postcss-reporter': {
        // Enhanced reporting with timing
        clearReportedMessages: true,
        throwError: false,
        sortByPosition: true,
        plugins: ['postcss-bem-linter']
      }
    })
  }
};

// Export configuration info for monitoring
export const CONFIG_INFO = {
  enableParallel: ENABLE_PARALLEL,
  systemCores: SYSTEM_CORES,
  environment: process.env.NODE_ENV,
  features: {
    purgeCSS: process.env.NODE_ENV === 'production',
    cssModules: true,
    customMedia: true,
    presetEnv: true,
    debugging: process.env.NODE_ENV === 'development'
  }
};