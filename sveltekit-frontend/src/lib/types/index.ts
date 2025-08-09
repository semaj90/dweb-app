// @ts-nocheck
// Core types stub for frontend services
export interface Database {
  [key: string]: any;
}

export interface API {
  [key: string]: any;
}

// Additional types can be added here as needed
export interface Config {
  [key: string]: any;
}

export interface Report {
  id: string;
  title: string;
  content: string;
  createdAt: string;
  updatedAt: string;
  status: 'draft' | 'completed' | 'archived';
  type: 'case' | 'evidence' | 'legal' | 'analysis';
  metadata?: Record<string, any>;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'prosecutor' | 'detective' | 'user';
  createdAt: string;
  updatedAt: string;
  preferences?: Record<string, any>;
}
