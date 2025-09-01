
// Canvas types for interactive features
export interface CanvasNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  type: 'case' | 'evidence' | 'person' | 'connection';
  data: any;
}

export interface CanvasConnection {
  id: string;
  sourceId: string;
  targetId: string;
  type: 'related' | 'involves' | 'contains';
  strength: number;
}

export interface InteractiveCanvasState {
  id?: string;
  nodes: CanvasNode[];
  connections: CanvasConnection[];
  viewport: {
    x: number;
    y: number;
    zoom: number;
  };
  animation?: string;
  frame?: number;
  fabricJSON?: string;
  metadata?: Record<string, any>;
}
