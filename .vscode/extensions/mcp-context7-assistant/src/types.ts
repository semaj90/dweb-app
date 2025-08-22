// Shared types for VS Code Extension and MCP integration
export interface VSCodeMCPContext {
  workspaceRoot: string;
  activeFiles: string[];
  currentFile?: string;
  errors: DiagnosticError[];
  userIntent: 'debugging' | 'feature-development' | 'optimization' | 'documentation';
  recentPrompts: string[];
  projectType: ProjectType;
  detectedStack: TechStack;
}

export type ProjectType = 
  | 'sveltekit-legal-ai'
  | 'react-nextjs' 
  | 'vue-nuxt'
  | 'cuda-gpu-computing'
  | 'electrical-embedded'
  | 'ml-ai-research'
  | 'systems-programming'
  | 'web-fullstack'
  | 'mobile-app'
  | 'desktop-native'
  | 'scientific-computing'
  | 'generic';

export interface TechStack {
  // Frontend Technologies
  frontend: Array<'react' | 'vue' | 'svelte' | 'angular' | 'html-css-js' | 'flutter' | 'react-native'>;
  
  // Backend Technologies  
  backend: Array<'node' | 'python' | 'rust' | 'go' | 'java' | 'cpp' | 'c' | 'csharp' | 'php'>;
  
  // Databases
  databases: Array<'postgresql' | 'mysql' | 'mongodb' | 'redis' | 'sqlite' | 'cassandra' | 'neo4j'>;
  
  // Cloud & DevOps
  cloud: Array<'aws' | 'gcp' | 'azure' | 'docker' | 'kubernetes' | 'terraform' | 'ansible'>;
  
  // AI/ML Technologies
  aiml: Array<'tensorflow' | 'pytorch' | 'scikit-learn' | 'opencv' | 'transformers' | 'langchain' | 'ollama'>;
  
  // GPU/CUDA Computing
  gpu: Array<'cuda' | 'opencl' | 'nvidia-toolkit' | 'tensorrt' | 'cupy' | 'numba' | 'thrust'>;
  
  // Embedded/Hardware
  embedded: Array<'arduino' | 'raspberry-pi' | 'esp32' | 'stm32' | 'fpga' | 'verilog' | 'vhdl' | 'bare-metal'>;
  
  // Systems Programming
  systems: Array<'kernel' | 'drivers' | 'real-time' | 'embedded-linux' | 'rtos' | 'bare-metal'>;
  
  // Scientific Computing
  scientific: Array<'numpy' | 'scipy' | 'matlab' | 'mathematica' | 'jupyter' | 'pandas' | 'r-lang'>;
  
  // Game Development
  gaming: Array<'unity' | 'unreal' | 'godot' | 'opengl' | 'vulkan' | 'directx'>;
  
  // Mobile Development
  mobile: Array<'android-native' | 'ios-native' | 'flutter' | 'react-native' | 'xamarin'>;
  
  // Web3/Blockchain
  web3: Array<'ethereum' | 'solidity' | 'web3js' | 'hardhat' | 'truffle'>;
}

export interface DiagnosticError {
  file: string;
  line: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
  source?: 'typescript' | 'eslint' | 'svelte' | 'other';
}

export interface AutoMCPSuggestion {
  tool: 'analyze-stack' | 'generate-best-practices' | 'suggest-integration' | 'get-library-docs' | 'resolve-library-id';
  confidence: number;
  reasoning: string;
  args: Record<string, any>;
  priority: 'high' | 'medium' | 'low';
  expectedOutput: string;
}

export interface MCPContextAnalysis {
  detectedStack: string[];
  currentErrors: DiagnosticError[];
  suggestedActions: AutoMCPSuggestion[];
  contextConfidence: number;
}

export interface MCPServerStatus {
  running: boolean;
  port: number;
  pid?: number;
  startTime?: Date;
  lastActivity?: Date;
}

export interface ExtensionConfig {
  serverPort: number;
  autoStart: boolean;
  contextDetection: boolean;
  suggestionConfidenceThreshold: number;
  maxSuggestions: number;
  logLevel: 'error' | 'warn' | 'info' | 'debug';
}

// VS Code specific interfaces
export interface StatusBarState {
  status: 'ready' | 'analyzing' | 'executing' | 'error' | 'offline';
  message: string;
  tooltip?: string;
}

export interface MCPToolResult {
  success: boolean;
  data?: unknown;
  error?: string;
  executionTime?: number;
}