/**
 * YoRHa Legal AI Assistant - Interactive JavaScript
 * Handles chat functionality, API calls, and UI interactions
 */

class YoRHaAssistant {
    constructor() {
        this.isMinimized = true;
        this.isHidden = false;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.messages = [];
        this.isTyping = false;
        this.connectionStatus = 'connected';
        
        // API Configuration
        this.apiEndpoints = {
            gemma3: 'http://localhost:11434/api/generate',
            enhanced: 'http://localhost:8000/v1/chat/completions',
            health: 'http://localhost:8000/health',
            qdrant: 'http://localhost:6333'
        };
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkSystemStatus();
        this.startPeriodicHealthCheck();
    }

    initializeElements() {
        // Main elements
        this.assistant = document.getElementById('yorha-assistant');
        this.toggleBtn = document.getElementById('assistant-toggle');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        
        // Header elements
        this.assistantHeader = document.getElementById('assistant-header');
        this.minimizeBtn = document.getElementById('minimize-btn');
        this.closeBtn = document.getElementById('close-btn');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.connectionStatusEl = document.getElementById('connection-status');
        this.notificationBadge = document.getElementById('notification-badge');
        
        // Demo page elements
        this.queryBtns = document.querySelectorAll('.query-btn');
        this.endpointElements = document.querySelectorAll('.endpoint');
    }

    setupEventListeners() {
        // Toggle button
        this.toggleBtn.addEventListener('click', () => this.toggleAssistant());
        
        // Header controls
        this.minimizeBtn.addEventListener('click', () => this.toggleMinimize());
        this.closeBtn.addEventListener('click', () => this.hideAssistant());
        
        // Chat input
        this.chatInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));
        this.chatInput.addEventListener('input', () => this.handleInputChange());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Dragging functionality
        this.assistantHeader.addEventListener('mousedown', (e) => this.startDragging(e));
        document.addEventListener('mousemove', (e) => this.handleDragging(e));
        document.addEventListener('mouseup', () => this.stopDragging());
        
        // Sample query buttons
        this.queryBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const query = btn.getAttribute('data-query');
                this.insertSampleQuery(query);
            });
        });
        
        // Auto-resize textarea
        this.chatInput.addEventListener('input', () => this.autoResizeTextarea());
    }

    toggleAssistant() {
        if (this.isHidden) {
            this.showAssistant();
        } else if (this.isMinimized) {
            this.expandAssistant();
        } else {
            this.minimizeAssistant();
        }
        this.hideNotificationBadge();
    }

    showAssistant() {
        this.isHidden = false;
        this.assistant.classList.remove('hidden');
        this.expandAssistant();
    }

    hideAssistant() {
        this.isHidden = true;
        this.assistant.classList.add('hidden');
    }

    expandAssistant() {
        this.isMinimized = false;
        this.assistant.classList.remove('minimized');
        setTimeout(() => {
            this.chatInput.focus();
            this.scrollToBottom();
        }, 300);
    }

    minimizeAssistant() {
        this.isMinimized = true;
        this.assistant.classList.add('minimized');
    }

    toggleMinimize() {
        if (this.isMinimized) {
            this.expandAssistant();
        } else {
            this.minimizeAssistant();
        }
    }

    startDragging(e) {
        if (e.target.closest('.control-btn')) return;
        
        this.isDragging = true;
        this.assistant.classList.add('dragging');
        
        const rect = this.assistant.getBoundingClientRect();
        this.dragOffset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        
        e.preventDefault();
    }

    handleDragging(e) {
        if (!this.isDragging) return;
        
        const x = e.clientX - this.dragOffset.x;
        const y = e.clientY - this.dragOffset.y;
        
        // Constrain to viewport
        const maxX = window.innerWidth - this.assistant.offsetWidth;
        const maxY = window.innerHeight - this.assistant.offsetHeight;
        
        const constrainedX = Math.max(0, Math.min(x, maxX));
        const constrainedY = Math.max(0, Math.min(y, maxY));
        
        this.assistant.style.left = `${constrainedX}px`;
        this.assistant.style.top = `${constrainedY}px`;
        this.assistant.style.right = 'auto';
        this.assistant.style.bottom = 'auto';
        
        e.preventDefault();
    }

    stopDragging() {
        if (!this.isDragging) return;
        
        this.isDragging = false;
        this.assistant.classList.remove('dragging');
    }

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    handleInputChange() {
        const hasText = this.chatInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText;
    }

    autoResizeTextarea() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
    }

    insertSampleQuery(query) {
        this.chatInput.value = query;
        this.handleInputChange();
        this.autoResizeTextarea();
        
        if (!this.isMinimized && !this.isHidden) {
            this.chatInput.focus();
        } else {
            this.showAssistant();
            setTimeout(() => this.chatInput.focus(), 300);
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage('user', message);
        this.chatInput.value = '';
        this.handleInputChange();
        this.autoResizeTextarea();

        // Show typing indicator
        this.setTypingState(true);

        try {
            // Try enhanced API first, fallback to direct Ollama
            let response;
            try {
                response = await this.callEnhancedAPI(message);
            } catch (error) {
                console.warn('Enhanced API failed, trying direct Ollama:', error);
                response = await this.callGemma3Direct(message);
            }

            this.addMessage('ai', response);
        } catch (error) {
            console.error('All APIs failed:', error);
            this.addErrorMessage('Sorry, I\'m having trouble connecting to the AI service. Please try again.');
        } finally {
            this.setTypingState(false);
        }
    }

    async callEnhancedAPI(message) {
        const requestBody = {
            messages: [
                {
                    role: "user",
                    content: message
                }
            ],
            max_tokens: 512,
            temperature: 0.1
        };

        const response = await fetch(this.apiEndpoints.enhanced, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody),
            signal: AbortSignal.timeout(60000)
        });

        if (!response.ok) {
            throw new Error(`Enhanced API error: ${response.status}`);
        }

        const data = await response.json();
        return data.response || data.choices?.[0]?.message?.content || 'No response received';
    }

    async callGemma3Direct(message) {
        const requestBody = {
            model: "gemma3-legal:latest",
            prompt: `<start_of_turn>user\n${message}<end_of_turn>\n<start_of_turn>model\n`,
            stream: false,
            options: {
                temperature: 0.1,
                num_predict: 512,
                top_k: 40,
                top_p: 0.9,
                stop: ["<start_of_turn>", "<end_of_turn>"]
            }
        };

        const response = await fetch(this.apiEndpoints.gemma3, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody),
            signal: AbortSignal.timeout(90000)
        });

        if (!response.ok) {
            throw new Error(`Gemma3 API error: ${response.status}`);
        }

        const data = await response.json();
        return data.response?.trim() || 'No response received';
    }

    addMessage(role, content) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}-message`;

        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        
        if (role === 'ai') {
            // Add typewriter effect for AI messages
            this.typewriterEffect(contentElement, content);
        } else {
            contentElement.innerHTML = this.formatMessageContent(content);
        }

        const timeElement = document.createElement('div');
        timeElement.className = 'message-time';
        timeElement.textContent = this.formatTime(new Date());

        messageElement.appendChild(contentElement);
        messageElement.appendChild(timeElement);

        this.chatMessages.appendChild(messageElement);
        this.scrollToBottom();

        // Store message
        this.messages.push({ role, content, timestamp: new Date() });
    }

    addErrorMessage(errorText) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.innerHTML = `
            <p>${errorText}</p>
            <button class="retry-btn" onclick="assistantInstance.retryLastMessage()">Retry</button>
        `;
        this.chatMessages.appendChild(errorElement);
        this.scrollToBottom();
    }

    retryLastMessage() {
        // Remove error message
        const errorMsg = this.chatMessages.querySelector('.error-message:last-child');
        if (errorMsg) errorMsg.remove();

        // Get last user message and retry
        const lastUserMessage = this.messages.filter(m => m.role === 'user').pop();
        if (lastUserMessage) {
            this.chatInput.value = lastUserMessage.content;
            this.sendMessage();
        }
    }

    typewriterEffect(element, text) {
        element.innerHTML = '';
        let index = 0;
        
        const typeChar = () => {
            if (index < text.length) {
                element.innerHTML += text.charAt(index);
                index++;
                setTimeout(typeChar, 20);
            } else {
                element.innerHTML = this.formatMessageContent(text);
            }
            this.scrollToBottom();
        };
        
        typeChar();
    }

    formatMessageContent(content) {
        // Convert markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/```(.*?)```/gs, '<code>$1</code>');
    }

    formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    setTypingState(isTyping) {
        this.isTyping = isTyping;
        
        if (isTyping) {
            this.typingIndicator.classList.add('active');
            this.sendBtn.disabled = true;
        } else {
            this.typingIndicator.classList.remove('active');
            this.handleInputChange();
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 50);
    }

    showNotificationBadge() {
        this.notificationBadge.style.display = 'flex';
    }

    hideNotificationBadge() {
        this.notificationBadge.style.display = 'none';
    }

    updateConnectionStatus(status) {
        this.connectionStatus = status;
        this.connectionStatusEl.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
        this.connectionStatusEl.className = `connection-status ${status === 'connected' ? '' : 'disconnected'}`;
    }

    async checkSystemStatus() {
        const endpoints = [
            { name: 'Ollama API', url: this.apiEndpoints.gemma3.replace('/api/generate', '/api/version') },
            { name: 'Enhanced API', url: this.apiEndpoints.health },
            { name: 'Qdrant Vector', url: this.apiEndpoints.qdrant }
        ];

        for (let i = 0; i < this.endpointElements.length && i < endpoints.length; i++) {
            const element = this.endpointElements[i];
            const endpoint = endpoints[i];
            const statusEl = element.querySelector('.endpoint-status');

            try {
                const response = await fetch(endpoint.url, { 
                    method: 'GET',
                    signal: AbortSignal.timeout(5000)
                });
                
                if (response.ok) {
                    statusEl.className = 'endpoint-status online';
                } else {
                    statusEl.className = 'endpoint-status offline';
                }
            } catch (error) {
                statusEl.className = 'endpoint-status offline';
            }
        }
    }

    startPeriodicHealthCheck() {
        // Check system health every 30 seconds
        setInterval(() => {
            this.checkSystemStatus();
        }, 30000);

        // Check connection status every 10 seconds
        setInterval(async () => {
            try {
                const response = await fetch(this.apiEndpoints.health, {
                    method: 'GET',
                    signal: AbortSignal.timeout(5000)
                });
                
                if (response.ok) {
                    this.updateConnectionStatus('connected');
                } else {
                    this.updateConnectionStatus('disconnected');
                }
            } catch (error) {
                this.updateConnectionStatus('disconnected');
            }
        }, 10000);
    }

    // Public methods for external access
    sendQuery(query) {
        this.insertSampleQuery(query);
    }

    show() {
        this.showAssistant();
    }

    hide() {
        this.hideAssistant();
    }
}

// Initialize the assistant when DOM is loaded
let assistantInstance;

document.addEventListener('DOMContentLoaded', () => {
    assistantInstance = new YoRHaAssistant();
    
    // Make it globally accessible
    window.yorhaAssistant = assistantInstance;
    
    console.log('🚀 YoRHa Legal AI Assistant initialized');
    console.log('📱 Access via: window.yorhaAssistant');
});

// Add some demo functionality
document.addEventListener('DOMContentLoaded', () => {
    // Add click handlers for demo features
    const demoCards = document.querySelectorAll('.feature-card');
    demoCards.forEach(card => {
        card.addEventListener('click', () => {
            const cardTitle = card.querySelector('h3').textContent;
            let query = '';
            
            switch (cardTitle) {
                case 'Contract Analysis':
                    query = 'Can you help me analyze a software license agreement? What key clauses should I review?';
                    break;
                case 'Liability Assessment':
                    query = 'I need help understanding liability limitations in my contract. What should I look for?';
                    break;
                case 'Document Search':
                    query = 'How can I search for specific clauses across multiple legal documents?';
                    break;
                case 'Legal Consultation':
                    query = 'I have questions about contract termination clauses and notice requirements.';
                    break;
                default:
                    query = 'Hello! I need help with legal document analysis.';
            }
            
            if (assistantInstance) {
                assistantInstance.sendQuery(query);
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to toggle assistant
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (assistantInstance) {
                assistantInstance.toggleAssistant();
            }
        }
        
        // Escape to minimize assistant
        if (e.key === 'Escape' && assistantInstance && !assistantInstance.isMinimized) {
            assistantInstance.minimizeAssistant();
        }
    });
});

// Add some utility functions
window.YoRHaUtils = {
    formatLegalText: (text) => {
        return text
            .replace(/\b(liability|indemnification|termination|compliance)\b/gi, '<span style="color: var(--yorha-primary);">$1</span>')
            .replace(/\b(shall|must|will|may)\b/gi, '<span style="color: var(--yorha-accent);">$1</span>');
    },
    
    highlightLegalTerms: (element) => {
        if (element && element.innerHTML) {
            element.innerHTML = window.YoRHaUtils.formatLegalText(element.innerHTML);
        }
    }
};
