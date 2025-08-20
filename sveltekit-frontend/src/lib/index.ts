/**
 * Legal AI Platform - Comprehensive Library Exports
 * SvelteKit 2 + Svelte 5 + TypeScript
 * 
 * Centralized export file for all components, services, stores, and utilities
 */

// ===== CORE UI COMPONENTS =====
export { default as Button } from './components/ui/Button.svelte';
export { default as Card } from './components/ui/Card.svelte';
export { default as Input } from './components/ui/Input.svelte';
export { default as Label } from './components/ui/Label.svelte';
export { default as Modal } from './components/ui/Modal.svelte';
export { default as Textarea } from './components/ui/textarea/Textarea.svelte';

// Enhanced UI Components (Bits UI Integration)
export { default as EnhancedButton } from './components/ui/enhanced/Button.svelte';
export { default as EnhancedCard } from './components/ui/enhanced/Card.svelte';
export { default as EnhancedInput } from './components/ui/enhanced/Input.svelte';

// Dialog and Modal Components
export { default as Dialog } from './components/ui/dialog/Dialog.svelte';
export { default as DialogContent } from './components/ui/dialog/DialogContent.svelte';
export { default as DialogHeader } from './components/ui/dialog/DialogHeader.svelte';
export { default as DialogTitle } from './components/ui/dialog/DialogTitle.svelte';

// Form Components
export { default as Checkbox } from './components/ui/checkbox/Checkbox.svelte';
export { default as Select } from './components/ui/select/Select.svelte';
export { default as SelectContent } from './components/ui/select/SelectContent.svelte';
export { default as SelectItem } from './components/ui/select/SelectItem.svelte';

// ===== AI COMPONENTS =====
export { default as AIChat } from './components/ai/AIChat.svelte';
export { default as AIAssistant } from './components/ai/AiAssistant.svelte';
export { default as EnhancedAIAssistant } from './components/ai/EnhancedAIAssistant.svelte';
export { default as AIProcessingDashboard } from './components/ai/AIProcessingDashboard.svelte';
export { default as ChatInterface } from './components/ai/ChatInterface.svelte';
export { default as ChatMessage } from './components/ai/ChatMessage.svelte';

// RAG and Vector Search Components
export { default as EnhancedRAGInterface } from './components/enhanced-rag/EnhancedRAGInterface.svelte';
export { default as VectorSearchWidget } from './components/vector/VectorSearchWidget.svelte';
export { default as VectorIntelligenceDemo } from './components/vector/VectorIntelligenceDemo.svelte';

// ===== LEGAL COMPONENTS =====
export { default as LegalCaseManager } from './components/LegalCaseManager.svelte';
export { default as CaseCard } from './components/cases/CaseCard.svelte';
export { default as CaseForm } from './components/forms/CaseForm.svelte';
export { default as EnhancedCaseForm } from './components/forms/EnhancedCaseForm.svelte';

// Evidence Components
export { default as EvidenceProcessor } from './components/evidence/EvidenceProcessor.svelte';
export { default as EvidenceCard } from './components/evidence/EvidenceCard.svelte';
export { default as EvidenceUploader } from './components/EvidenceUploader.svelte';
export { default as EvidenceAnalysisForm } from './components/EvidenceAnalysisForm.svelte';

// Document Components
export { default as DocumentUploadForm } from './components/DocumentUploadForm.svelte';
export { default as EnhancedDocumentUploadForm } from './components/forms/EnhancedDocumentUploadForm.svelte';
export { default as LegalDocumentEditor } from './components/editor/LegalDocumentEditor.svelte';

// ===== AUTHENTICATION COMPONENTS =====
export { default as AuthForm } from './components/auth/AuthForm.svelte';
export { default as LoginModal } from './components/auth/LoginModal.svelte';
export { default as RegisterModal } from './components/auth/RegisterModal.svelte';

// ===== CANVAS AND VISUAL COMPONENTS =====
export { default as CanvasEditor } from './components/CanvasEditor.svelte';
export { default as EnhancedCanvasEditor } from './components/EnhancedCanvasEditor.svelte';
export { default as EvidenceCanvas } from './components/ai/EvidenceCanvas.svelte';
export { default as DetectiveBoard } from './components/detective/DetectiveBoard.svelte';

// ===== UPLOAD AND FILE COMPONENTS =====
export { default as AdvancedFileUpload } from './components/upload/AdvancedFileUpload.svelte';
export { default as FileUploadForm } from './components/upload/FileUploadForm.svelte';
export { default as FileUploadProgress } from './components/upload/FileUploadProgress.svelte';
export { default as UploadArea } from './components/UploadArea.svelte';

// ===== NOTIFICATION AND FEEDBACK =====
export { default as NotificationContainer } from './components/notifications/EnhancedNotificationContainer.svelte';
export { default as ErrorBoundary } from './components/ErrorBoundary.svelte';
export { default as LoadingSpinner } from './components/LoadingSpinner.svelte';
export { default as ProgressIndicator } from './components/ProgressIndicator.svelte';

// ===== SYSTEM MANAGEMENT COMPONENTS =====
export { default as KeyboardShortcutsPanel } from './components/KeyboardShortcutsPanel.svelte';
export { default as LoggingDashboard } from './components/LoggingDashboard.svelte';

// ===== YORHA THEME COMPONENTS =====
export { default as YoRHaCommandCenter } from './components/yorha/YoRHaCommandCenter.svelte';
export { default as YoRHaDialog } from './components/yorha/YoRHaDialog.svelte';
export { default as YoRHaNavigation } from './components/yorha/YoRHaNavigation.svelte';
export { default as YoRHaTerminal } from './components/yorha/YoRHaTerminal.svelte';

// ===== STORES =====
export { default as authStore } from './stores/auth-store.svelte';
export { aiStore } from './stores/ai';
export { evidenceStore } from './stores/evidence';
export { enhancedRAGStore } from './stores/enhanced-rag-store';
export { realtimeStore } from './stores/realtime';
export { chatStore } from './stores/chatStore';

// ===== SERVICES =====
export { aiAutoTaggingService } from './services/ai-auto-tagging-service';
export { ollamaService } from './services/ollama-service';
export { documentApiService } from './services/documentApi';
export { multiLayerCache } from './services/multiLayerCache';
export { unifiedDocumentProcessor } from './services/unified-document-processor';

// Context7 and MCP Services
export { context7Service } from './services/context7-mcp-integration';
export { comprehensiveCachingService } from './services/comprehensive-caching-service';

// Keyboard Shortcuts and Remote Control
export { keyboardShortcutsService, shortcuts, shortcutCategories, remoteCommands, isRemoteConnected } from './services/keyboard-shortcuts-service';

// Logging Aggregation
export { loggingService, logEntries, logStats, debug, info, warn, error, fatal } from './services/logging-aggregation-service';

// ===== UTILITIES =====
export { cn } from './utils';
export { formatDate, formatCurrency, formatFileSize } from './utils/formatters';
export { validateEmail, validateRequired, validateFile } from './utils/validators';
export { generateId, createSlug, debounce, throttle } from './utils/helpers';
export { uploadFile, processFile, extractText } from './utils/file-processing';

// ===== TYPES =====
export type { CommonProps } from './types/common-props';
export type { User, Session, AuthState } from './types/auth';
export type { Case, Evidence, Document } from './types/legal';
export type { AIResponse, RAGResult, VectorSearchResult } from './types/ai';
export type { UploadResult, FileProcessingStatus } from './types/upload';

// ===== CONFIGURATION =====
export { defaultConfig } from './config/defaults';
export { apiEndpoints } from './config/api-endpoints';
export { themeConfig } from './config/theme';

// ===== MACHINE DEFINITIONS (XState) =====
export { agentShellMachine } from './machines/agentShellMachine';
export { canvasSystemMachine } from './machines/canvasSystem';
export { evidenceProcessingMachine } from './state/evidenceProcessingMachine';

// ===== HOOKS AND ACTIONS =====
export { useAuth } from './hooks/useAuth';
export { useAI } from './hooks/useAI';
export { useUpload } from './hooks/useUpload';
export { tooltip } from './ui/actions/tooltip';

// ===== OPTIMIZATION AND PERFORMANCE =====
export { comprehensiveOrchestrator } from './optimization/comprehensive-orchestrator';
export { memoryEfficientExtension } from './optimization/memory-efficient-extension';
export { context7MCPIntegration } from './optimization/context7-mcp-integration';

// ===== DATABASE AND ORM =====
export { enhancedSchema } from './database/enhanced-schema';
export { enhancedVectorOperations } from './server/db/enhanced-vector-operations';

// ===== WASM AND GPU =====
export { gpuWasmInit } from './wasm/gpu-wasm-init';

// ===== RE-EXPORTS FROM UI LIBRARIES =====
// Bits UI Components
export {
  Dialog as BitsDialog,
  Button as BitsButton,
  Select as BitsSelect,
  AlertDialog as BitsAlertDialog
} from 'bits-ui';

// Melt UI Utilities
export { createDialog, createSelect, createToaster } from '@melt-ui/svelte';

// Lucide Icons (commonly used)
export {
  Search,
  Upload,
  Download,
  User,
  Settings,
  Bell,
  MessageSquare,
  FileText,
  Database,
  Cpu,
  Zap,
  Shield,
  AlertCircle,
  CheckCircle,
  X,
  Plus,
  Minus,
  Edit,
  Trash2,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  Menu,
  Home,
  Folder,
  File,
  Image,
  Video,
  Music,
  Archive,
  Link,
  ExternalLink,
  Mail,
  Phone,
  Calendar,
  Clock,
  MapPin,
  Globe,
  Wifi,
  WifiOff,
  Battery,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Square,
  SkipBack,
  SkipForward,
  Repeat,
  Shuffle,
  Heart,
  Star,
  Bookmark,
  Share,
  Copy,
  Clipboard,
  RotateCcw,
  RotateCw,
  RefreshCw,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  ArrowDownLeft,
  TrendingUp,
  TrendingDown,
  BarChart,
  PieChart,
  Activity,
  Monitor,
  Smartphone,
  Tablet,
  Laptop,
  Server,
  HardDrive,
  Wifi as WifiIcon,
  Bluetooth,
  Usb,
  MousePointer,
  Keyboard,
  Printer,
  Camera,
  Mic,
  MicOff,
  Video as VideoIcon,
  VideoOff,
  Lock,
  Unlock,
  Key,
  UserCheck,
  UserX,
  Users,
  UserPlus,
  UserMinus,
  Crown,
  Award,
  Gift,
  Package,
  ShoppingCart,
  ShoppingBag,
  CreditCard,
  DollarSign,
  PoundSterling,
  Euro,
  Banknote,
  Calculator,
  Target,
  Crosshair,
  Focus,
  Maximize,
  Minimize,
  Square as SquareIcon,
  Circle,
  Triangle,
  Hexagon,
  Octagon,
  Smile,
  Frown,
  Meh,
  Laughing,
  Angry,
  Surprised,
  Confused,
  Sleeping,
  ThumbsUp,
  ThumbsDown,
  Flag,
  Bookmark as BookmarkIcon,
  Tag,
  Hash,
  AtSign,
  Percent,
  Ampersand,
  Asterisk,
  Slash,
  Backslash,
  Quote,
  Semicolon,
  Colon,
  Dot,
  Comma,
  Exclamation,
  Question,
  Tilde,
  Underscore,
  Hyphen,
  Equal,
  NotEqual,
  GreaterThan,
  LessThan,
  GreaterEqual,
  LessEqual,
  PlusCircle,
  MinusCircle,
  XCircle,
  CheckCircle2,
  AlertCircle as AlertCircleIcon,
  InfoIcon,
  HelpCircle,
  RadioButton,
  ToggleLeft,
  ToggleRight,
  Power,
  PowerOff,
  Zap as ZapIcon,
  Flash,
  Flame,
  Sun,
  Moon,
  CloudRain,
  Cloud,
  CloudSnow,
  CloudLightning,
  Umbrella,
  Wind,
  Thermometer,
  Droplets,
  Waves,
  Mountain,
  Tree,
  Flower,
  Leaf,
  Sprout,
  Seedling
} from 'lucide-svelte';

// ===== VERSION INFO =====
export const VERSION = '2.0.0';
export const BUILD_DATE = new Date().toISOString();
export const FRAMEWORK_INFO = {
  sveltekit: '2.x',
  svelte: '5.x',
  typescript: '5.x',
  vite: '5.x'
};

// ===== FEATURE FLAGS =====
export const FEATURES = {
  GPU_ACCELERATION: true,
  VECTOR_SEARCH: true,
  REAL_TIME_CHAT: true,
  CONTEXT7_INTEGRATION: true,
  MULTI_PROTOCOL_API: true,
  YORHA_THEME: true,
  MCP_INTEGRATION: true,
  WASM_SUPPORT: true,
  WEBGPU_SUPPORT: true,
  CUDA_SUPPORT: true
} as const;

// ===== DEVELOPMENT UTILITIES =====
export const DEV_TOOLS = {
  COMPONENT_COUNT: 392,
  ROUTE_COUNT: 82,
  API_ENDPOINT_COUNT: 145,
  STORE_COUNT: 8,
  SERVICE_COUNT: 12
} as const;

// Default export for convenience
export default {
  VERSION,
  BUILD_DATE,
  FRAMEWORK_INFO,
  FEATURES,
  DEV_TOOLS
};