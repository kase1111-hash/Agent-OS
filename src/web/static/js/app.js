/**
 * Agent OS Web Interface
 * Main JavaScript application
 */

class AgentOS {
    constructor() {
        this.ws = null;
        this.conversationId = null;
        this.currentView = 'chat';
        this.currentUser = null;
        this.isAuthenticated = false;

        // Security utilities
        this.crypto = window.secureCrypto || new SecureCrypto();
        this.secureStorage = window.secureStorage || new SecureStorage(this.crypto);
        this.redactor = window.redactor || new SensitiveDataRedactor();

        // Debug settings
        this.debugMode = localStorage.getItem('debugMode') === 'true';
        this.debugLogs = [];
        this.networkLogs = [];
        this.maxLogs = 500;
        this.debugFilter = 'all';
        this.currentDebugTab = 'logs';
        this.debugPanelMinimized = false;

        // Settings (loaded asynchronously for encrypted values)
        this.settings = {
            debug_mode: this.debugMode,
            verbose_logging: localStorage.getItem('verboseLogging') === 'true',
            auto_reconnect: localStorage.getItem('autoReconnect') !== 'false',
            sound_notifications: localStorage.getItem('soundNotifications') === 'true',
            dark_theme: localStorage.getItem('darkTheme') !== 'false',
            ollama_endpoint: 'http://localhost:11434',
            llama_cpp_endpoint: 'http://localhost:8080',
            default_model: localStorage.getItem('defaultModel') || 'llama3.2:3b'
        };

        // Initialize with encrypted settings
        this._initSecureSettings();
        this.init();
    }

    /**
     * Initialize secure settings asynchronously
     */
    async _initSecureSettings() {
        try {
            // Migrate any existing plain text sensitive values
            await this.secureStorage.migrateToEncrypted();

            // Load encrypted settings
            const ollamaEndpoint = await this.secureStorage.getItem('ollamaEndpoint');
            const llamaCppEndpoint = await this.secureStorage.getItem('llamaCppEndpoint');

            if (ollamaEndpoint) this.settings.ollama_endpoint = ollamaEndpoint;
            if (llamaCppEndpoint) this.settings.llama_cpp_endpoint = llamaCppEndpoint;

            // Refresh UI if settings page is visible
            this.loadSettings();
        } catch (error) {
            console.error('Failed to load secure settings:', error);
        }
    }

    init() {
        this.setupNavigation();
        this.setupChat();
        this.setupModal();
        this.setupDebugPanel();
        this.loadSettings();
        this.checkAuthStatus().then(() => {
            this.connectWebSocket();
            this.loadInitialData();
        });

        // Start dreaming status polling (every 5 seconds)
        this.startDreamingPoll();
        this.setupClickOutside();
        this.interceptConsole();
        this.interceptFetch();

        // Initialize debug mode if previously enabled
        if (this.debugMode) {
            this.toggleDebugMode(true, false);
        }
    }

    // =========================================================================
    // Authentication
    // =========================================================================

    async checkAuthStatus() {
        try {
            const response = await fetch('/api/auth/status');
            const data = await response.json();

            if (data.authenticated && data.user) {
                this.setAuthenticatedUser(data.user);
            } else {
                this.setUnauthenticated();
            }
        } catch (error) {
            console.error('Failed to check auth status:', error);
            this.setUnauthenticated();
        }
    }

    setAuthenticatedUser(user) {
        this.currentUser = user;
        this.isAuthenticated = true;

        // Update UI
        document.getElementById('auth-buttons').style.display = 'none';
        document.getElementById('user-menu').style.display = 'flex';

        // Set user info
        const displayName = user.display_name || user.username;
        document.getElementById('user-display-name').textContent = displayName;
        document.getElementById('user-avatar').textContent = displayName.charAt(0).toUpperCase();
        document.getElementById('dropdown-username').textContent = user.username;
        document.getElementById('dropdown-role').textContent = user.role;
    }

    setUnauthenticated() {
        this.currentUser = null;
        this.isAuthenticated = false;

        // Update UI
        document.getElementById('auth-buttons').style.display = 'flex';
        document.getElementById('user-menu').style.display = 'none';
    }

    toggleUserDropdown() {
        const dropdown = document.getElementById('user-dropdown');
        dropdown.classList.toggle('active');
    }

    setupClickOutside() {
        document.addEventListener('click', (e) => {
            const userMenu = document.getElementById('user-menu');
            const dropdown = document.getElementById('user-dropdown');

            if (userMenu && dropdown && !userMenu.contains(e.target)) {
                dropdown.classList.remove('active');
            }
        });
    }

    showLoginModal() {
        this.showModal('Sign In', `
            <form id="login-form" onsubmit="app.handleLogin(event)">
                <div class="form-group">
                    <label for="login-username">Username or Email</label>
                    <input type="text" id="login-username" placeholder="Enter your username or email" required autofocus>
                </div>
                <div class="form-group">
                    <label for="login-password">Password</label>
                    <input type="password" id="login-password" placeholder="Enter your password" required>
                </div>
                <div class="form-group form-checkbox">
                    <label>
                        <input type="checkbox" id="login-remember">
                        Remember me for 30 days
                    </label>
                </div>
                <div id="login-error" class="form-error" style="display: none;"></div>
            </form>
        `, `
            <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
            <button class="btn btn-primary" onclick="app.handleLogin(event)">Sign In</button>
        `);
    }

    showRegisterModal() {
        this.showModal('Create Account', `
            <form id="register-form" onsubmit="app.handleRegister(event)">
                <div class="form-group">
                    <label for="register-username">Username *</label>
                    <input type="text" id="register-username" placeholder="Choose a username (3+ characters)" required minlength="3" maxlength="50" autofocus>
                </div>
                <div class="form-group">
                    <label for="register-email">Email (optional)</label>
                    <input type="email" id="register-email" placeholder="Enter your email address">
                </div>
                <div class="form-group">
                    <label for="register-display-name">Display Name (optional)</label>
                    <input type="text" id="register-display-name" placeholder="How should we call you?">
                </div>
                <div class="form-group">
                    <label for="register-password">Password *</label>
                    <input type="password" id="register-password" placeholder="Choose a password (6+ characters)" required minlength="6">
                </div>
                <div class="form-group">
                    <label for="register-password-confirm">Confirm Password *</label>
                    <input type="password" id="register-password-confirm" placeholder="Confirm your password" required>
                </div>
                <div id="register-error" class="form-error" style="display: none;"></div>
            </form>
        `, `
            <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
            <button class="btn btn-primary" onclick="app.handleRegister(event)">Create Account</button>
        `);
    }

    async handleLogin(event) {
        if (event) event.preventDefault();

        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;
        const rememberMe = document.getElementById('login-remember').checked;
        const errorDiv = document.getElementById('login-error');

        if (!username || !password) {
            this.showFormError(errorDiv, 'Please enter both username and password');
            return;
        }

        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: username,
                    password: password,
                    remember_me: rememberMe
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.setAuthenticatedUser(data.user);
                this.hideModal();
                this.showNotification('Welcome back, ' + (data.user.display_name || data.user.username) + '!', 'success');
            } else {
                this.showFormError(errorDiv, data.detail || 'Login failed');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showFormError(errorDiv, 'Login failed. Please try again.');
        }
    }

    async handleRegister(event) {
        if (event) event.preventDefault();

        const username = document.getElementById('register-username').value.trim();
        const email = document.getElementById('register-email').value.trim();
        const displayName = document.getElementById('register-display-name').value.trim();
        const password = document.getElementById('register-password').value;
        const passwordConfirm = document.getElementById('register-password-confirm').value;
        const errorDiv = document.getElementById('register-error');

        // Validation
        if (!username || username.length < 3) {
            this.showFormError(errorDiv, 'Username must be at least 3 characters');
            return;
        }

        if (!password || password.length < 6) {
            this.showFormError(errorDiv, 'Password must be at least 6 characters');
            return;
        }

        if (password !== passwordConfirm) {
            this.showFormError(errorDiv, 'Passwords do not match');
            return;
        }

        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: username,
                    password: password,
                    email: email || null,
                    display_name: displayName || null
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.setAuthenticatedUser(data.user);
                this.hideModal();
                this.showNotification('Account created! Welcome, ' + (data.user.display_name || data.user.username) + '!', 'success');
            } else {
                this.showFormError(errorDiv, data.detail || 'Registration failed');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.showFormError(errorDiv, 'Registration failed. Please try again.');
        }
    }

    async logout() {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
        } catch (error) {
            console.error('Logout error:', error);
        }

        this.setUnauthenticated();
        document.getElementById('user-dropdown').classList.remove('active');
        this.showNotification('You have been signed out', 'info');
    }

    showProfileModal() {
        if (!this.currentUser) return;

        document.getElementById('user-dropdown').classList.remove('active');

        this.showModal('Edit Profile', `
            <form id="profile-form" onsubmit="app.handleProfileUpdate(event)">
                <div class="form-group">
                    <label for="profile-username">Username</label>
                    <input type="text" id="profile-username" value="${this.escapeHtml(this.currentUser.username)}" readonly disabled>
                    <small style="color: var(--text-muted);">Username cannot be changed</small>
                </div>
                <div class="form-group">
                    <label for="profile-display-name">Display Name</label>
                    <input type="text" id="profile-display-name" value="${this.escapeHtml(this.currentUser.display_name || '')}" placeholder="Enter your display name">
                </div>
                <div class="form-group">
                    <label for="profile-email">Email</label>
                    <input type="email" id="profile-email" value="${this.escapeHtml(this.currentUser.email || '')}" placeholder="Enter your email address">
                </div>
                <div id="profile-error" class="form-error" style="display: none;"></div>
            </form>
        `, `
            <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
            <button class="btn btn-primary" onclick="app.handleProfileUpdate(event)">Save Changes</button>
        `);
    }

    async handleProfileUpdate(event) {
        if (event) event.preventDefault();

        const displayName = document.getElementById('profile-display-name').value.trim();
        const email = document.getElementById('profile-email').value.trim();
        const errorDiv = document.getElementById('profile-error');

        try {
            const response = await fetch('/api/auth/profile', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    display_name: displayName || null,
                    email: email || null
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                if (data.user) {
                    this.setAuthenticatedUser(data.user);
                }
                this.hideModal();
                this.showNotification('Profile updated successfully', 'success');
            } else {
                this.showFormError(errorDiv, data.detail || 'Update failed');
            }
        } catch (error) {
            console.error('Profile update error:', error);
            this.showFormError(errorDiv, 'Update failed. Please try again.');
        }
    }

    showChangePasswordModal() {
        document.getElementById('user-dropdown').classList.remove('active');

        this.showModal('Change Password', `
            <form id="password-form" onsubmit="app.handlePasswordChange(event)">
                <div class="form-group">
                    <label for="current-password">Current Password *</label>
                    <input type="password" id="current-password" placeholder="Enter your current password" required>
                </div>
                <div class="form-group">
                    <label for="new-password">New Password *</label>
                    <input type="password" id="new-password" placeholder="Enter your new password (6+ characters)" required minlength="6">
                </div>
                <div class="form-group">
                    <label for="confirm-new-password">Confirm New Password *</label>
                    <input type="password" id="confirm-new-password" placeholder="Confirm your new password" required>
                </div>
                <div id="password-error" class="form-error" style="display: none;"></div>
            </form>
        `, `
            <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
            <button class="btn btn-primary" onclick="app.handlePasswordChange(event)">Change Password</button>
        `);
    }

    async handlePasswordChange(event) {
        if (event) event.preventDefault();

        const currentPassword = document.getElementById('current-password').value;
        const newPassword = document.getElementById('new-password').value;
        const confirmNewPassword = document.getElementById('confirm-new-password').value;
        const errorDiv = document.getElementById('password-error');

        if (newPassword.length < 6) {
            this.showFormError(errorDiv, 'New password must be at least 6 characters');
            return;
        }

        if (newPassword !== confirmNewPassword) {
            this.showFormError(errorDiv, 'New passwords do not match');
            return;
        }

        try {
            const response = await fetch('/api/auth/change-password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.hideModal();
                this.showNotification('Password changed successfully', 'success');
            } else {
                this.showFormError(errorDiv, data.detail || 'Password change failed');
            }
        } catch (error) {
            console.error('Password change error:', error);
            this.showFormError(errorDiv, 'Password change failed. Please try again.');
        }
    }

    async showSessionsModal() {
        document.getElementById('user-dropdown').classList.remove('active');

        try {
            const response = await fetch('/api/auth/sessions');
            const data = await response.json();

            const sessions = data.sessions || [];

            this.showModal('Active Sessions', `
                <div class="sessions-list">
                    ${sessions.length === 0 ? '<p style="color: var(--text-muted);">No active sessions</p>' :
                        sessions.map(session => `
                            <div class="session-item">
                                <div class="session-info">
                                    <div class="session-device">
                                        <svg viewBox="0 0 24 24" width="20" height="20"><path fill="currentColor" d="M4 6h18V4H4c-1.1 0-2 .9-2 2v11H0v3h14v-3H4V6zm19 2h-6c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h6c.55 0 1-.45 1-1V9c0-.55-.45-1-1-1zm-1 9h-4v-7h4v7z"/></svg>
                                        <span>${session.ip_address || 'Unknown device'}</span>
                                    </div>
                                    <div class="session-meta">
                                        <span>Last active: ${new Date(session.last_activity).toLocaleString()}</span>
                                    </div>
                                </div>
                                <button class="btn btn-danger btn-small" onclick="app.revokeSession('${session.session_id}')">Revoke</button>
                            </div>
                        `).join('')
                    }
                </div>
            `, `
                <button class="btn btn-secondary" onclick="app.hideModal()">Close</button>
                ${sessions.length > 1 ? '<button class="btn btn-danger" onclick="app.logoutAll()">Sign Out All</button>' : ''}
            `);
        } catch (error) {
            console.error('Failed to load sessions:', error);
            this.showError('Failed to load sessions');
        }
    }

    async revokeSession(sessionId) {
        try {
            await fetch(`/api/auth/sessions/${sessionId}`, { method: 'DELETE' });
            this.showSessionsModal(); // Refresh
            this.showNotification('Session revoked', 'success');
        } catch (error) {
            this.showError('Failed to revoke session');
        }
    }

    async logoutAll() {
        try {
            await fetch('/api/auth/logout-all', { method: 'POST' });
            this.setUnauthenticated();
            this.hideModal();
            this.showNotification('Signed out from all devices', 'info');
        } catch (error) {
            this.showError('Failed to sign out from all devices');
        }
    }

    showFormError(element, message) {
        if (element) {
            element.textContent = message;
            element.style.display = 'block';
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${this.escapeHtml(message)}</span>
            <button onclick="this.parentElement.remove()">&times;</button>
        `;

        // Add to page
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            document.body.appendChild(container);
        }

        container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    // Navigation
    setupNavigation() {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                const view = e.target.dataset.view;
                this.switchView(view);
            });
        });
    }

    switchView(view) {
        // Update nav
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.dataset.view === view);
        });

        // Update view
        document.querySelectorAll('.view').forEach(v => {
            v.classList.toggle('active', v.id === `${view}-view`);
        });

        this.currentView = view;

        // Load view-specific data
        switch (view) {
            case 'images':
                this.loadImages();
                break;
            case 'agents':
                this.loadAgents();
                break;
            case 'constitution':
                this.loadConstitution();
                break;
            case 'memory':
                this.loadMemory();
                break;
            case 'contracts':
                this.loadContracts();
                break;
            case 'system':
                this.loadSystem();
                break;
        }
    }

    // WebSocket
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/chat/ws`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };

            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(connected) {
        const status = document.getElementById('connection-status');
        if (connected) {
            status.textContent = 'Connected';
            status.classList.remove('disconnected');
            status.classList.add('connected');
        } else {
            status.textContent = 'Disconnected';
            status.classList.remove('connected');
            status.classList.add('disconnected');
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'connected':
                this.conversationId = data.conversation_id;
                break;
            case 'response':
                this.appendMessage(data.message);
                break;
            case 'history':
                this.renderChatHistory(data.messages);
                break;
            case 'error':
                this.showError(data.message);
                break;
        }
    }

    // Chat
    setupChat() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const newChatBtn = document.getElementById('new-chat-btn');

        sendBtn.addEventListener('click', () => this.sendMessage());

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 150) + 'px';
        });

        newChatBtn.addEventListener('click', () => this.startNewChat());
    }

    sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();

        if (!message) return;

        // Add user message to UI
        this.appendMessage({
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });

        // Send via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'message',
                content: message
            }));
        } else {
            // Fallback to REST API
            this.sendMessageRest(message);
        }

        input.value = '';
        input.style.height = 'auto';
    }

    async sendMessageRest(message) {
        try {
            const response = await fetch('/api/chat/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.conversationId
                })
            });
            const data = await response.json();
            this.conversationId = data.conversation_id;
            this.appendMessage(data.message);
        } catch (error) {
            this.showError('Failed to send message');
        }
    }

    appendMessage(message) {
        const container = document.getElementById('chat-messages');
        const welcome = container.querySelector('.welcome-message');
        if (welcome) {
            welcome.remove();
        }

        const div = document.createElement('div');
        div.className = `message ${message.role}`;

        const time = new Date(message.timestamp).toLocaleTimeString();

        div.innerHTML = `
            <div class="message-content">${this.escapeHtml(message.content)}</div>
            <div class="message-time">${time}</div>
        `;

        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    renderChatHistory(messages) {
        const container = document.getElementById('chat-messages');
        container.innerHTML = '';
        messages.forEach(msg => this.appendMessage(msg));
    }

    startNewChat() {
        this.conversationId = null;
        document.getElementById('chat-messages').innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to Agent OS</h2>
                <p>Start a conversation with your AI assistant.</p>
            </div>
        `;

        // Reconnect to get new conversation ID
        if (this.ws) {
            this.ws.close();
        }
        this.connectWebSocket();
    }

    // Agents
    async loadAgents() {
        try {
            const [agents, stats] = await Promise.all([
                fetch('/api/agents/').then(r => r.json()),
                fetch('/api/agents/stats/overview').then(r => r.json())
            ]);

            this.renderAgentsStats(stats);
            this.renderAgentsList(agents);
        } catch (error) {
            console.error('Failed to load agents:', error);
        }
    }

    renderAgentsStats(stats) {
        const container = document.getElementById('agents-stats');
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_agents}</div>
                <div class="stat-label">Total Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.total_requests}</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.success_rate.toFixed(1)}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.status_distribution?.active || 0}</div>
                <div class="stat-label">Active Agents</div>
            </div>
        `;
    }

    renderAgentsList(agents) {
        const container = document.getElementById('agents-list');
        container.innerHTML = agents.map(agent => `
            <div class="agent-card">
                <div class="agent-header">
                    <span class="agent-name">${agent.name}</span>
                    <span class="agent-status ${agent.status}">${agent.status}</span>
                </div>
                <div class="agent-description">${agent.description || 'No description'}</div>
                <div class="agent-metrics">
                    <div class="agent-metric">
                        <span>Requests:</span>
                        <span>${agent.requests_total}</span>
                    </div>
                </div>
                <div class="agent-actions">
                    <button class="btn btn-secondary" onclick="app.viewAgent('${agent.name}')">View</button>
                    ${agent.status === 'active'
                        ? `<button class="btn btn-danger" onclick="app.stopAgent('${agent.name}')">Stop</button>`
                        : `<button class="btn btn-primary" onclick="app.startAgent('${agent.name}')">Start</button>`
                    }
                </div>
            </div>
        `).join('');
    }

    async startAgent(name) {
        try {
            await fetch(`/api/agents/${name}/start`, { method: 'POST' });
            this.loadAgents();
        } catch (error) {
            this.showError('Failed to start agent');
        }
    }

    async stopAgent(name) {
        try {
            await fetch(`/api/agents/${name}/stop`, { method: 'POST' });
            this.loadAgents();
        } catch (error) {
            this.showError('Failed to stop agent');
        }
    }

    async viewAgent(name) {
        try {
            const agent = await fetch(`/api/agents/${name}`).then(r => r.json());
            this.showModal('Agent Details', `
                <div class="form-group">
                    <label>Name</label>
                    <input type="text" value="${agent.name}" readonly>
                </div>
                <div class="form-group">
                    <label>Description</label>
                    <textarea readonly>${agent.description}</textarea>
                </div>
                <div class="form-group">
                    <label>Status</label>
                    <input type="text" value="${agent.status}" readonly>
                </div>
                <div class="form-group">
                    <label>Capabilities</label>
                    <div>${agent.capabilities.map(c => `<span class="tag">${c.name}</span>`).join(' ')}</div>
                </div>
            `);
        } catch (error) {
            this.showError('Failed to load agent details');
        }
    }

    // Constitution
    async loadConstitution() {
        try {
            const [sections, rules] = await Promise.all([
                fetch('/api/constitution/sections').then(r => r.json()),
                fetch('/api/constitution/rules').then(r => r.json())
            ]);

            this.renderSections(sections);
            this.renderRules(rules);
        } catch (error) {
            console.error('Failed to load constitution:', error);
        }
    }

    renderSections(sections) {
        const container = document.getElementById('sections-list');
        container.innerHTML = `
            <div class="section-item active" data-section="all" onclick="app.loadConstitution()">
                All Rules
            </div>
        ` + sections.map(section => `
            <div class="section-item" data-section="${section.id}" onclick="app.filterRulesBySection('${section.id}')">
                ${section.title}
                <span class="section-count">${section.rules ? section.rules.length : 0}</span>
            </div>
        `).join('');
    }

    renderRules(rules) {
        const container = document.getElementById('rules-list');
        container.innerHTML = rules.map(rule => `
            <div class="rule-card ${rule.rule_type}">
                <div class="rule-header">
                    <span class="rule-id">${rule.id}</span>
                    <span class="rule-type">${rule.rule_type}</span>
                    <span class="rule-authority">${rule.authority}</span>
                </div>
                <div class="rule-content">${this.escapeHtml(rule.content)}</div>
                <div class="rule-keywords">
                    ${rule.keywords.map(k => `<span class="keyword">${k}</span>`).join('')}
                </div>
                <div class="rule-actions">
                    ${!rule.is_immutable ? `
                        <button class="btn btn-secondary btn-small" onclick="app.editRule('${rule.id}')">Edit</button>
                        <button class="btn btn-danger btn-small" onclick="app.deleteRule('${rule.id}')">Delete</button>
                    ` : '<span style="color: var(--text-muted); font-size: 0.8rem;">Immutable</span>'}
                </div>
            </div>
        `).join('');
    }

    async filterRulesBySection(sectionId) {
        // Highlight selected section
        document.querySelectorAll('.section-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.section === sectionId) {
                item.classList.add('active');
            }
        });

        // Reload and filter rules by section
        try {
            const sections = await fetch('/api/constitution/sections').then(r => r.json());
            const section = sections.find(s => s.id === sectionId);
            if (section && section.rules) {
                this.renderRules(section.rules);
            }
        } catch (error) {
            console.error('Failed to filter rules:', error);
        }
    }

    async showAddRuleModal() {
        this.showModal('Add New Rule', `
            <div class="form-group">
                <label>Rule Content *</label>
                <textarea id="rule-content" placeholder="Enter the rule text..." rows="3"></textarea>
            </div>
            <div class="form-group">
                <label>Rule Type *</label>
                <select id="rule-type">
                    <option value="prohibition">Prohibition - Things that MUST NOT be done</option>
                    <option value="permission">Permission - Things that MAY be done</option>
                    <option value="mandate">Mandate - Things that MUST be done</option>
                    <option value="escalation">Escalation - Things requiring confirmation</option>
                </select>
            </div>
            <div class="form-group">
                <label>Authority Level</label>
                <select id="rule-authority">
                    <option value="statutory">Statutory - User-defined rules</option>
                    <option value="agent">Agent - Agent-specific rules</option>
                </select>
                <small style="color: var(--text-muted);">Note: Supreme and Constitutional rules can only be set during the ceremony process.</small>
            </div>
            <div class="form-group">
                <label>Keywords (comma-separated)</label>
                <input type="text" id="rule-keywords" placeholder="keyword1, keyword2, keyword3">
            </div>
            <div class="form-group">
                <label>Agent Scope (optional)</label>
                <select id="rule-agent-scope">
                    <option value="">All agents</option>
                    <option value="whisper">Whisper (Orchestrator)</option>
                    <option value="smith">Smith (Guardian)</option>
                    <option value="seshat">Seshat (Memory)</option>
                    <option value="sage">Sage (Reasoner)</option>
                    <option value="quill">Quill (Writer)</option>
                    <option value="muse">Muse (Creative)</option>
                </select>
            </div>
        `, `
            <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
            <button class="btn btn-primary" onclick="app.confirmAddRule()">Add Rule</button>
        `);
    }

    async confirmAddRule() {
        const content = document.getElementById('rule-content').value.trim();
        const ruleType = document.getElementById('rule-type').value;
        const authority = document.getElementById('rule-authority').value;
        const keywordsInput = document.getElementById('rule-keywords').value;
        const agentScope = document.getElementById('rule-agent-scope').value;

        if (!content) {
            this.showError('Rule content is required');
            return;
        }

        const keywords = keywordsInput.split(',').map(k => k.trim()).filter(k => k);

        try {
            const response = await fetch('/api/constitution/rules', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: content,
                    rule_type: ruleType,
                    authority: authority,
                    keywords: keywords,
                    agent_scope: agentScope || null
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to create rule');
            }

            this.hideModal();
            this.loadConstitution();
        } catch (error) {
            this.showError(error.message || 'Failed to add rule');
        }
    }

    async editRule(ruleId) {
        try {
            const rule = await fetch(`/api/constitution/rules/${ruleId}`).then(r => r.json());

            this.showModal('Edit Rule', `
                <div class="form-group">
                    <label>Rule ID</label>
                    <input type="text" value="${rule.id}" readonly>
                </div>
                <div class="form-group">
                    <label>Rule Content *</label>
                    <textarea id="edit-rule-content" rows="3">${this.escapeHtml(rule.content)}</textarea>
                </div>
                <div class="form-group">
                    <label>Rule Type *</label>
                    <select id="edit-rule-type">
                        <option value="prohibition" ${rule.rule_type === 'prohibition' ? 'selected' : ''}>Prohibition</option>
                        <option value="permission" ${rule.rule_type === 'permission' ? 'selected' : ''}>Permission</option>
                        <option value="mandate" ${rule.rule_type === 'mandate' ? 'selected' : ''}>Mandate</option>
                        <option value="escalation" ${rule.rule_type === 'escalation' ? 'selected' : ''}>Escalation</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Keywords (comma-separated)</label>
                    <input type="text" id="edit-rule-keywords" value="${rule.keywords.join(', ')}">
                </div>
            `, `
                <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
                <button class="btn btn-primary" onclick="app.confirmEditRule('${ruleId}')">Save Changes</button>
            `);
        } catch (error) {
            this.showError('Failed to load rule');
        }
    }

    async confirmEditRule(ruleId) {
        const content = document.getElementById('edit-rule-content').value.trim();
        const ruleType = document.getElementById('edit-rule-type').value;
        const keywordsInput = document.getElementById('edit-rule-keywords').value;

        if (!content) {
            this.showError('Rule content is required');
            return;
        }

        const keywords = keywordsInput.split(',').map(k => k.trim()).filter(k => k);

        try {
            const response = await fetch(`/api/constitution/rules/${ruleId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: content,
                    rule_type: ruleType,
                    keywords: keywords
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update rule');
            }

            this.hideModal();
            this.loadConstitution();
        } catch (error) {
            this.showError(error.message || 'Failed to update rule');
        }
    }

    async deleteRule(ruleId) {
        if (!confirm('Are you sure you want to delete this rule?')) {
            return;
        }

        try {
            const response = await fetch(`/api/constitution/rules/${ruleId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete rule');
            }

            this.loadConstitution();
        } catch (error) {
            this.showError(error.message || 'Failed to delete rule');
        }
    }

    // Memory
    async loadMemory() {
        try {
            const [memories, stats] = await Promise.all([
                fetch('/api/memory/').then(r => r.json()),
                fetch('/api/memory/stats').then(r => r.json())
            ]);

            this.renderMemoryStats(stats);
            this.renderMemoryList(memories);
        } catch (error) {
            console.error('Failed to load memory:', error);
        }
    }

    renderMemoryStats(stats) {
        const container = document.getElementById('memory-stats');
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_entries}</div>
                <div class="stat-label">Total Entries</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(stats.total_size_bytes / 1024).toFixed(1)} KB</div>
                <div class="stat-label">Total Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.consent_rate.toFixed(0)}%</div>
                <div class="stat-label">Consent Rate</div>
            </div>
        `;
    }

    renderMemoryList(memories) {
        const container = document.getElementById('memory-list');
        container.innerHTML = memories.map(memory => `
            <div class="memory-card">
                <div class="memory-header">
                    <span class="memory-type ${memory.memory_type}">${memory.memory_type}</span>
                    <button class="btn btn-danger" onclick="app.deleteMemory('${memory.id}')">Delete</button>
                </div>
                <div class="memory-content">${this.escapeHtml(memory.content)}</div>
                <div class="memory-tags">
                    ${memory.tags.map(t => `<span class="tag">${t}</span>`).join('')}
                </div>
            </div>
        `).join('');
    }

    async deleteMemory(id) {
        if (!confirm('Are you sure you want to delete this memory?')) return;

        try {
            await fetch(`/api/memory/${id}`, { method: 'DELETE' });
            this.loadMemory();
        } catch (error) {
            this.showError('Failed to delete memory');
        }
    }

    async showAddMemoryModal() {
        this.showModal('Add New Memory', `
            <div class="form-group">
                <label>Memory Content *</label>
                <textarea id="memory-content" placeholder="Enter the memory content..." rows="4"></textarea>
            </div>
            <div class="form-group">
                <label>Memory Type</label>
                <select id="memory-type">
                    <option value="working">Working - Short-term operational memory</option>
                    <option value="long_term">Long Term - Persistent memory</option>
                    <option value="semantic">Semantic - Factual knowledge</option>
                    <option value="ephemeral">Ephemeral - Temporary, auto-deleted</option>
                </select>
            </div>
            <div class="form-group">
                <label>Tags (comma-separated)</label>
                <input type="text" id="memory-tags" placeholder="tag1, tag2, tag3">
            </div>
            <div class="form-group">
                <label>Retention (days, optional)</label>
                <input type="number" id="memory-retention" placeholder="Leave empty for default">
            </div>
            <div class="form-group">
                <label>
                    <input type="checkbox" id="memory-consent" checked>
                    I consent to storing this memory
                </label>
                <small style="color: var(--text-muted);">Required for memory storage per constitutional requirements.</small>
            </div>
        `, `
            <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
            <button class="btn btn-primary" onclick="app.confirmAddMemory()">Store Memory</button>
        `);
    }

    async confirmAddMemory() {
        const content = document.getElementById('memory-content').value.trim();
        const memoryType = document.getElementById('memory-type').value;
        const tagsInput = document.getElementById('memory-tags').value;
        const retentionInput = document.getElementById('memory-retention').value;
        const consent = document.getElementById('memory-consent').checked;

        if (!content) {
            this.showError('Memory content is required');
            return;
        }

        if (!consent) {
            this.showError('Consent is required to store memory');
            return;
        }

        const tags = tagsInput.split(',').map(t => t.trim()).filter(t => t);
        const retention = retentionInput ? parseInt(retentionInput) : null;

        try {
            const response = await fetch('/api/memory/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: content,
                    memory_type: memoryType,
                    tags: tags,
                    consent_given: consent,
                    retention_days: retention
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to store memory');
            }

            this.hideModal();
            this.loadMemory();
        } catch (error) {
            this.showError(error.message || 'Failed to store memory');
        }
    }

    // Contracts
    async loadContracts() {
        try {
            const [contracts, templates, stats] = await Promise.all([
                fetch('/api/contracts/').then(r => r.json()),
                fetch('/api/contracts/templates').then(r => r.json()),
                fetch('/api/contracts/stats').then(r => r.json())
            ]);

            this.renderContractsStats(stats);
            this.renderTemplates(templates);
            this.renderContracts(contracts);
            this.setupContractsTabs();
        } catch (error) {
            console.error('Failed to load contracts:', error);
        }
    }

    renderContractsStats(stats) {
        const container = document.getElementById('contracts-stats');
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_contracts}</div>
                <div class="stat-label">Total Contracts</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.active_contracts}</div>
                <div class="stat-label">Active</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.expired_contracts}</div>
                <div class="stat-label">Expired</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.revoked_contracts}</div>
                <div class="stat-label">Revoked</div>
            </div>
        `;
    }

    renderTemplates(templates) {
        const container = document.getElementById('templates-list');
        container.innerHTML = templates.map(template => `
            <div class="template-card" onclick="app.createFromTemplate('${template.id}')">
                <div class="template-name">${template.name}</div>
                <div class="template-type ${template.contract_type.toLowerCase()}">${template.contract_type}</div>
                <div class="template-description">${template.description}</div>
            </div>
        `).join('');
    }

    renderContracts(contracts) {
        const container = document.getElementById('contracts-list');

        if (contracts.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No contracts found</p>
                    <p style="color: var(--text-muted);">Create a contract using the templates on the left</p>
                </div>
            `;
            return;
        }

        container.innerHTML = contracts.map(contract => `
            <div class="contract-card ${contract.status.toLowerCase()}">
                <div class="contract-header">
                    <span class="contract-type ${contract.contract_type.toLowerCase()}">${contract.contract_type}</span>
                    <span class="contract-status ${contract.status.toLowerCase()}">${contract.status}</span>
                </div>
                <div class="contract-domains">
                    ${contract.domains.map(d => `<span class="domain-tag">${d}</span>`).join('')}
                </div>
                <div class="contract-meta">
                    <span>Created: ${new Date(contract.created_at).toLocaleDateString()}</span>
                    ${contract.expires_at
                        ? `<span>Expires: ${new Date(contract.expires_at).toLocaleDateString()}</span>`
                        : '<span>No expiration</span>'
                    }
                </div>
                <div class="contract-actions">
                    <button class="btn btn-secondary" onclick="app.viewContract('${contract.id}')">View</button>
                    ${contract.status === 'ACTIVE'
                        ? `<button class="btn btn-danger" onclick="app.revokeContract('${contract.id}')">Revoke</button>`
                        : ''
                    }
                </div>
            </div>
        `).join('');
    }

    setupContractsTabs() {
        document.querySelectorAll('.contracts-tab').forEach(tab => {
            tab.addEventListener('click', async (e) => {
                // Update active tab
                document.querySelectorAll('.contracts-tab').forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');

                // Filter contracts
                const status = e.target.dataset.status.toUpperCase();
                try {
                    const contracts = await fetch(`/api/contracts/?status=${status}`).then(r => r.json());
                    this.renderContracts(contracts);
                } catch (error) {
                    console.error('Failed to filter contracts:', error);
                }
            });
        });
    }

    async createFromTemplate(templateId) {
        try {
            const template = await fetch(`/api/contracts/templates/${templateId}`).then(r => r.json());

            this.showModal('Create Contract from Template', `
                <div class="template-preview">
                    <h4>${template.name}</h4>
                    <p>${template.description}</p>
                    <div class="template-info">
                        <div class="info-row">
                            <label>Contract Type:</label>
                            <span class="contract-type ${template.contract_type.toLowerCase()}">${template.contract_type}</span>
                        </div>
                        <div class="info-row">
                            <label>Recommended for:</label>
                            <span>${template.recommended_for}</span>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <label>Domains (comma-separated)</label>
                    <input type="text" id="contract-domains" value="${template.default_domains.join(', ')}" placeholder="coding, development">
                </div>
                <div class="form-group">
                    <label>Duration (days, leave empty for no expiration)</label>
                    <input type="number" id="contract-duration" value="${template.default_duration_days || ''}" placeholder="365">
                </div>
            `, `
                <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
                <button class="btn btn-primary" onclick="app.confirmCreateFromTemplate('${templateId}')">Create Contract</button>
            `);
        } catch (error) {
            this.showError('Failed to load template');
        }
    }

    async confirmCreateFromTemplate(templateId) {
        const domainsInput = document.getElementById('contract-domains');
        const durationInput = document.getElementById('contract-duration');

        const domains = domainsInput.value.split(',').map(d => d.trim()).filter(d => d);
        const duration = durationInput.value ? parseInt(durationInput.value) : null;

        try {
            await fetch('/api/contracts/from-template', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    template_id: templateId,
                    domains: domains,
                    duration_days: duration
                })
            });
            this.hideModal();
            this.loadContracts();
        } catch (error) {
            this.showError('Failed to create contract');
        }
    }

    async showCreateContractModal() {
        try {
            const types = await fetch('/api/contracts/types').then(r => r.json());

            this.showModal('Create New Contract', `
                <div class="form-group">
                    <label>Contract Type</label>
                    <select id="contract-type">
                        ${types.map(t => `
                            <option value="${t.name}" title="${t.description}">${t.name} - ${t.description}</option>
                        `).join('')}
                    </select>
                </div>
                <div class="form-group">
                    <label>Domains (comma-separated)</label>
                    <input type="text" id="contract-domains" placeholder="coding, development, work">
                </div>
                <div class="form-group">
                    <label>Duration (days, leave empty for no expiration)</label>
                    <input type="number" id="contract-duration" placeholder="365">
                </div>
                <div class="form-group">
                    <label>Description</label>
                    <textarea id="contract-description" placeholder="Describe the purpose of this contract..."></textarea>
                </div>
            `, `
                <button class="btn btn-secondary" onclick="app.hideModal()">Cancel</button>
                <button class="btn btn-primary" onclick="app.confirmCreateContract()">Create Contract</button>
            `);
        } catch (error) {
            this.showError('Failed to load contract types');
        }
    }

    async confirmCreateContract() {
        const type = document.getElementById('contract-type').value;
        const domainsInput = document.getElementById('contract-domains');
        const durationInput = document.getElementById('contract-duration');
        const descInput = document.getElementById('contract-description');

        const domains = domainsInput.value.split(',').map(d => d.trim()).filter(d => d);
        const duration = durationInput.value ? parseInt(durationInput.value) : null;

        try {
            await fetch('/api/contracts/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contract_type: type,
                    domains: domains,
                    duration_days: duration,
                    description: descInput.value
                })
            });
            this.hideModal();
            this.loadContracts();
        } catch (error) {
            this.showError('Failed to create contract');
        }
    }

    async viewContract(contractId) {
        try {
            const contract = await fetch(`/api/contracts/${contractId}`).then(r => r.json());

            this.showModal('Contract Details', `
                <div class="contract-details">
                    <div class="form-group">
                        <label>Contract ID</label>
                        <input type="text" value="${contract.id}" readonly>
                    </div>
                    <div class="form-group">
                        <label>Type</label>
                        <span class="contract-type ${contract.contract_type.toLowerCase()}">${contract.contract_type}</span>
                    </div>
                    <div class="form-group">
                        <label>Status</label>
                        <span class="contract-status ${contract.status.toLowerCase()}">${contract.status}</span>
                    </div>
                    <div class="form-group">
                        <label>Domains</label>
                        <div>${contract.domains.map(d => `<span class="domain-tag">${d}</span>`).join(' ')}</div>
                    </div>
                    <div class="form-group">
                        <label>Description</label>
                        <textarea readonly>${contract.description || 'No description'}</textarea>
                    </div>
                    <div class="form-group">
                        <label>Created</label>
                        <input type="text" value="${new Date(contract.created_at).toLocaleString()}" readonly>
                    </div>
                    <div class="form-group">
                        <label>Expires</label>
                        <input type="text" value="${contract.expires_at ? new Date(contract.expires_at).toLocaleString() : 'Never'}" readonly>
                    </div>
                </div>
            `);
        } catch (error) {
            this.showError('Failed to load contract details');
        }
    }

    async revokeContract(contractId) {
        if (!confirm('Are you sure you want to revoke this contract? This action cannot be undone.')) {
            return;
        }

        try {
            await fetch(`/api/contracts/${contractId}/revoke`, { method: 'POST' });
            this.loadContracts();
        } catch (error) {
            this.showError('Failed to revoke contract');
        }
    }

    // System
    async loadSystem() {
        try {
            const [info, health, settings] = await Promise.all([
                fetch('/api/system/info').then(r => r.json()),
                fetch('/api/system/health').then(r => r.json()),
                fetch('/api/system/settings').then(r => r.json())
            ]);

            this.renderSystemInfo(info);
            this.renderSystemHealth(health);
            this.renderSettings(settings);
        } catch (error) {
            console.error('Failed to load system:', error);
        }
    }

    renderSystemInfo(info) {
        const container = document.getElementById('system-info');
        container.innerHTML = `
            <h3>System Information</h3>
            <div class="info-item">
                <span class="info-label">Version</span>
                <span>${info.version}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Platform</span>
                <span>${info.platform}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Python</span>
                <span>${info.python_version.split(' ')[0]}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Started</span>
                <span>${new Date(info.started_at).toLocaleString()}</span>
            </div>
        `;
    }

    renderSystemHealth(health) {
        const container = document.getElementById('system-health');
        container.innerHTML = `
            <h3>Component Health</h3>
            ${health.map(h => `
                <div class="health-item">
                    <span class="health-dot ${h.status}"></span>
                    <span>${h.name}</span>
                    <span style="margin-left: auto; color: var(--text-muted);">
                        ${h.latency_ms ? h.latency_ms.toFixed(1) + 'ms' : ''}
                    </span>
                </div>
            `).join('')}
        `;
    }

    renderSettings(settings) {
        const container = document.getElementById('settings-list');
        container.innerHTML = settings.map(setting => `
            <div class="setting-item">
                <div class="setting-key">${setting.key}</div>
                <div class="setting-description">${setting.description}</div>
                <div class="setting-value">
                    ${this.renderSettingInput(setting)}
                </div>
            </div>
        `).join('');
    }

    renderSettingInput(setting) {
        if (setting.data_type === 'boolean') {
            return `<input type="checkbox" ${setting.value ? 'checked' : ''}
                    onchange="app.updateSetting('${setting.key}', this.checked)">`;
        } else if (setting.data_type === 'number') {
            return `<input type="number" value="${setting.value}"
                    onchange="app.updateSetting('${setting.key}', parseInt(this.value))">`;
        } else {
            return `<input type="text" value="${setting.value}"
                    onchange="app.updateSetting('${setting.key}', this.value)">`;
        }
    }

    async updateSetting(key, value) {
        try {
            await fetch(`/api/system/settings/${key}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ value })
            });
        } catch (error) {
            this.showError('Failed to update setting');
        }
    }

    // Images
    async loadImages() {
        try {
            const [stats, gallery] = await Promise.all([
                fetch('/api/images/stats').then(r => r.json()),
                fetch('/api/images/gallery').then(r => r.json())
            ]);

            this.renderImagesStats(stats);
            this.renderImageGallery(gallery);
        } catch (error) {
            console.error('Failed to load images:', error);
        }
    }

    renderImagesStats(stats) {
        const container = document.getElementById('images-stats');
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_jobs}</div>
                <div class="stat-label">Total Jobs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.completed_jobs}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.total_images}</div>
                <div class="stat-label">Images Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.pending_jobs}</div>
                <div class="stat-label">Pending</div>
            </div>
        `;
    }

    renderImageGallery(images) {
        const container = document.getElementById('images-gallery');

        if (images.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No images generated yet</p>
                    <p style="color: var(--text-muted);">Use the form on the left to generate your first image</p>
                </div>
            `;
            return;
        }

        container.innerHTML = images.map(image => `
            <div class="gallery-item" onclick="app.viewImage('${image.id}')">
                <img src="${image.thumbnail_url}" alt="${this.escapeHtml(image.prompt)}" loading="lazy">
                <div class="gallery-overlay">
                    <span class="gallery-prompt">${this.escapeHtml(image.prompt.substring(0, 50))}${image.prompt.length > 50 ? '...' : ''}</span>
                    <div class="gallery-meta">
                        <span>${image.width}x${image.height}</span>
                        <span>${image.model}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async generateImage(event) {
        if (event) event.preventDefault();

        const prompt = document.getElementById('image-prompt').value.trim();
        const negativePrompt = document.getElementById('image-negative-prompt').value.trim();
        const model = document.getElementById('image-model').value;
        const width = parseInt(document.getElementById('image-width').value);
        const height = parseInt(document.getElementById('image-height').value);
        const steps = parseInt(document.getElementById('image-steps').value);
        const guidance = parseFloat(document.getElementById('image-guidance').value);
        const seedInput = document.getElementById('image-seed').value;
        const numImages = parseInt(document.getElementById('image-count').value);

        if (!prompt) {
            this.showError('Please enter a prompt');
            return;
        }

        const generateBtn = document.getElementById('generate-btn');
        const progressDiv = document.getElementById('generation-progress');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');

        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        progressDiv.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Starting generation...';

        try {
            const response = await fetch('/api/images/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    negative_prompt: negativePrompt || null,
                    model: model,
                    width: width,
                    height: height,
                    steps: steps,
                    guidance_scale: guidance,
                    seed: seedInput ? parseInt(seedInput) : null,
                    num_images: numImages
                })
            });

            const job = await response.json();

            if (!response.ok) {
                throw new Error(job.detail || 'Generation failed');
            }

            // Poll for completion
            await this.pollGenerationJob(job.id, progressFill, progressText);

            // Refresh gallery
            this.loadImages();

            // Reset form
            document.getElementById('image-prompt').value = '';
            document.getElementById('image-negative-prompt').value = '';

            this.showNotification('Image generated successfully!', 'success');

        } catch (error) {
            console.error('Generation error:', error);
            this.showError(error.message || 'Failed to generate image');
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Image';
            progressDiv.style.display = 'none';
        }
    }

    async pollGenerationJob(jobId, progressFill, progressText) {
        const maxAttempts = 120; // 2 minutes max
        let attempts = 0;

        while (attempts < maxAttempts) {
            const response = await fetch(`/api/images/generate/${jobId}`);
            const job = await response.json();

            if (job.status === 'completed') {
                progressFill.style.width = '100%';
                progressText.textContent = 'Complete!';
                return job;
            } else if (job.status === 'failed') {
                throw new Error(job.error || 'Generation failed');
            } else if (job.status === 'processing') {
                // Estimate progress based on attempts
                const estimatedProgress = Math.min((attempts / 30) * 100, 90);
                progressFill.style.width = `${estimatedProgress}%`;
                progressText.textContent = 'Processing...';
            }

            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
        }

        throw new Error('Generation timed out');
    }

    async viewImage(imageId) {
        try {
            // Find image in gallery
            const galleryResponse = await fetch('/api/images/gallery');
            const gallery = await galleryResponse.json();
            const image = gallery.find(img => img.id === imageId);

            if (!image) {
                this.showError('Image not found');
                return;
            }

            this.showModal('Image Details', `
                <div class="image-viewer">
                    <img src="${image.full_url}" alt="${this.escapeHtml(image.prompt)}" style="max-width: 100%; max-height: 60vh; object-fit: contain;">
                </div>
                <div class="image-details" style="margin-top: 1rem;">
                    <div class="form-group">
                        <label>Prompt</label>
                        <textarea readonly style="height: auto;">${this.escapeHtml(image.prompt)}</textarea>
                    </div>
                    ${image.negative_prompt ? `
                        <div class="form-group">
                            <label>Negative Prompt</label>
                            <textarea readonly style="height: auto;">${this.escapeHtml(image.negative_prompt)}</textarea>
                        </div>
                    ` : ''}
                    <div class="form-row" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                        <div class="form-group">
                            <label>Model</label>
                            <input type="text" value="${image.model}" readonly>
                        </div>
                        <div class="form-group">
                            <label>Size</label>
                            <input type="text" value="${image.width}x${image.height}" readonly>
                        </div>
                        <div class="form-group">
                            <label>Seed</label>
                            <input type="text" value="${image.seed}" readonly>
                        </div>
                    </div>
                    <div class="form-row" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                        <div class="form-group">
                            <label>Steps</label>
                            <input type="text" value="${image.steps}" readonly>
                        </div>
                        <div class="form-group">
                            <label>Guidance Scale</label>
                            <input type="text" value="${image.guidance_scale}" readonly>
                        </div>
                    </div>
                </div>
            `, `
                <button class="btn btn-secondary" onclick="app.hideModal()">Close</button>
                <button class="btn btn-primary" onclick="app.downloadImage('${imageId}')">Download</button>
                <button class="btn btn-danger" onclick="app.deleteImage('${imageId}')">Delete</button>
            `);
        } catch (error) {
            console.error('Failed to view image:', error);
            this.showError('Failed to load image details');
        }
    }

    async downloadImage(imageId) {
        try {
            const response = await fetch(`/api/images/image/${imageId}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `generated-${imageId}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            this.showError('Failed to download image');
        }
    }

    async deleteImage(imageId) {
        if (!confirm('Are you sure you want to delete this image?')) {
            return;
        }

        try {
            await fetch(`/api/images/gallery/${imageId}`, { method: 'DELETE' });
            this.hideModal();
            this.loadImages();
            this.showNotification('Image deleted', 'success');
        } catch (error) {
            this.showError('Failed to delete image');
        }
    }

    // =========================================================================
    // Debug Panel
    // =========================================================================

    setupDebugPanel() {
        const panel = document.getElementById('debug-panel');
        if (!panel) return;

        // Setup resize handle
        const resizeHandle = panel.querySelector('.debug-resize-handle');
        if (resizeHandle) {
            let startY, startHeight;

            resizeHandle.addEventListener('mousedown', (e) => {
                startY = e.clientY;
                startHeight = panel.offsetHeight;
                document.addEventListener('mousemove', resize);
                document.addEventListener('mouseup', stopResize);
            });

            const resize = (e) => {
                const newHeight = startHeight - (e.clientY - startY);
                if (newHeight >= 100 && newHeight <= window.innerHeight * 0.8) {
                    panel.style.height = newHeight + 'px';
                }
            };

            const stopResize = () => {
                document.removeEventListener('mousemove', resize);
                document.removeEventListener('mouseup', stopResize);
            };
        }
    }

    interceptConsole() {
        const originalLog = console.log;
        const originalWarn = console.warn;
        const originalError = console.error;

        console.log = (...args) => {
            this.addDebugLog('info', args.map(a => this.formatLogArg(a)).join(' '));
            originalLog.apply(console, args);
        };

        console.warn = (...args) => {
            this.addDebugLog('warn', args.map(a => this.formatLogArg(a)).join(' '));
            originalWarn.apply(console, args);
        };

        console.error = (...args) => {
            this.addDebugLog('error', args.map(a => this.formatLogArg(a)).join(' '));
            originalError.apply(console, args);
        };
    }

    interceptFetch() {
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            const startTime = performance.now();
            const url = typeof args[0] === 'string' ? args[0] : args[0].url;
            const method = args[1]?.method || 'GET';

            try {
                const response = await originalFetch.apply(window, args);
                const duration = Math.round(performance.now() - startTime);

                this.addNetworkLog({
                    method,
                    url,
                    status: response.status,
                    duration,
                    success: response.ok
                });

                if (this.settings.verbose_logging) {
                    this.addDebugLog('network', `${method} ${url} - ${response.status} (${duration}ms)`);
                }

                return response;
            } catch (error) {
                const duration = Math.round(performance.now() - startTime);
                this.addNetworkLog({
                    method,
                    url,
                    status: 'ERR',
                    duration,
                    success: false,
                    error: error.message
                });
                this.addDebugLog('error', `Network error: ${method} ${url} - ${error.message}`);
                throw error;
            }
        };
    }

    formatLogArg(arg) {
        if (typeof arg === 'object') {
            try {
                return JSON.stringify(arg, null, 2);
            } catch {
                return String(arg);
            }
        }
        return String(arg);
    }

    addDebugLog(level, message) {
        // Redact sensitive data before logging
        const safeMessage = this.redactor ? this.redactor.redact(message) : message;

        const entry = {
            time: new Date().toLocaleTimeString(),
            level,
            message: safeMessage
        };

        this.debugLogs.unshift(entry);
        if (this.debugLogs.length > this.maxLogs) {
            this.debugLogs.pop();
        }

        if (this.debugMode && this.currentDebugTab === 'logs') {
            this.renderDebugLogs();
        }
    }

    addNetworkLog(entry) {
        entry.time = new Date().toLocaleTimeString();

        // Redact sensitive data from URL and any error messages
        if (this.redactor) {
            entry.url = this.redactor.redactUrl(entry.url);
            if (entry.error) {
                entry.error = this.redactor.redact(entry.error);
            }
        }

        this.networkLogs.unshift(entry);
        if (this.networkLogs.length > this.maxLogs) {
            this.networkLogs.pop();
        }

        if (this.debugMode && this.currentDebugTab === 'network') {
            this.renderNetworkLogs();
        }
    }

    toggleDebugMode(enabled, save = true) {
        this.debugMode = enabled;
        if (save) {
            localStorage.setItem('debugMode', enabled);
        }

        const panel = document.getElementById('debug-panel');
        const checkbox = document.getElementById('setting-debug-mode');

        if (panel) {
            panel.style.display = enabled ? 'flex' : 'none';
        }

        if (checkbox) {
            checkbox.checked = enabled;
        }

        if (enabled) {
            this.addDebugLog('info', 'Debug mode enabled');
            this.renderDebugLogs();
            this.updateDebugState();
            this.updateDebugPerformance();
        }
    }

    minimizeDebugPanel() {
        const panel = document.getElementById('debug-panel');
        if (panel) {
            this.debugPanelMinimized = !this.debugPanelMinimized;
            panel.classList.toggle('minimized', this.debugPanelMinimized);
        }
    }

    switchDebugTab(tab) {
        this.currentDebugTab = tab;

        document.querySelectorAll('.debug-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tab);
        });

        document.querySelectorAll('.debug-tab-content').forEach(c => {
            c.classList.toggle('active', c.id === `debug-${tab}`);
        });

        switch (tab) {
            case 'logs':
                this.renderDebugLogs();
                break;
            case 'network':
                this.renderNetworkLogs();
                break;
            case 'state':
                this.updateDebugState();
                break;
            case 'performance':
                this.updateDebugPerformance();
                break;
        }
    }

    filterDebugLogs(filter) {
        this.debugFilter = filter;
        this.renderDebugLogs();
    }

    renderDebugLogs() {
        const container = document.getElementById('debug-log-entries');
        if (!container) return;

        const filteredLogs = this.debugFilter === 'all'
            ? this.debugLogs
            : this.debugLogs.filter(log => log.level === this.debugFilter);

        container.innerHTML = filteredLogs.slice(0, 100).map(log => `
            <div class="debug-log-entry">
                <span class="debug-log-time">${log.time}</span>
                <span class="debug-log-level ${log.level}">${log.level}</span>
                <span class="debug-log-message">${this.escapeHtml(log.message)}</span>
            </div>
        `).join('') || '<div style="color: var(--text-muted); padding: 1rem;">No logs yet</div>';
    }

    renderNetworkLogs() {
        const container = document.getElementById('debug-network-entries');
        if (!container) return;

        container.innerHTML = this.networkLogs.slice(0, 50).map(log => `
            <div class="debug-network-entry">
                <span class="debug-network-method ${log.method}">${log.method}</span>
                <span class="debug-network-url" title="${this.escapeHtml(log.url)}">${this.escapeHtml(log.url)}</span>
                <span class="debug-network-status ${log.success ? 'success' : 'error'}">${log.status}</span>
                <span class="debug-network-time">${log.duration}ms</span>
            </div>
        `).join('') || '<div style="color: var(--text-muted); padding: 1rem;">No network requests yet</div>';
    }

    updateDebugState() {
        const container = document.getElementById('debug-state-view');
        if (!container) return;

        // Redact sensitive data from settings before display
        const safeSettings = this.redactor ? this.redactor.redactObject({...this.settings}) : this.settings;

        const state = {
            currentView: this.currentView,
            isAuthenticated: this.isAuthenticated,
            currentUser: this.currentUser?.username || 'none',
            wsConnected: this.ws?.readyState === WebSocket.OPEN,
            conversationId: this.conversationId || 'none',
            debugMode: this.debugMode,
            encryptionEnabled: this.crypto?.isSupported || false,
            settings: safeSettings
        };

        container.innerHTML = Object.entries(state).map(([key, value]) => `
            <div class="debug-state-item">
                <div class="debug-state-key">${key}</div>
                <div class="debug-state-value">${typeof value === 'object' ? JSON.stringify(value, null, 2) : value}</div>
            </div>
        `).join('');
    }

    updateDebugPerformance() {
        const container = document.getElementById('debug-performance-view');
        if (!container) return;

        const perf = performance.getEntriesByType('navigation')[0];
        const memory = performance.memory || {};

        container.innerHTML = `
            <div class="debug-performance-metric">
                <span>Page Load Time</span>
                <span>${perf ? Math.round(perf.loadEventEnd - perf.startTime) : 'N/A'}ms</span>
            </div>
            <div class="debug-performance-metric">
                <span>DOM Content Loaded</span>
                <span>${perf ? Math.round(perf.domContentLoadedEventEnd - perf.startTime) : 'N/A'}ms</span>
            </div>
            <div class="debug-performance-metric">
                <span>JS Heap Size</span>
                <span>${memory.usedJSHeapSize ? (memory.usedJSHeapSize / 1024 / 1024).toFixed(2) + ' MB' : 'N/A'}</span>
            </div>
            <div class="debug-performance-metric">
                <span>Debug Logs</span>
                <span>${this.debugLogs.length}</span>
            </div>
            <div class="debug-performance-metric">
                <span>Network Requests</span>
                <span>${this.networkLogs.length}</span>
            </div>
            <div class="debug-performance-metric">
                <span>WebSocket State</span>
                <span>${this.ws ? ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][this.ws.readyState] : 'N/A'}</span>
            </div>
        `;
    }

    clearDebugLogs() {
        this.debugLogs = [];
        this.networkLogs = [];
        this.renderDebugLogs();
        this.renderNetworkLogs();
        this.addDebugLog('info', 'Logs cleared');
    }

    exportDebugLogs() {
        // Redact sensitive data before export
        const safeSettings = this.redactor ? this.redactor.redactObject({...this.settings}) : this.settings;

        const data = {
            timestamp: new Date().toISOString(),
            logs: this.debugLogs, // Already redacted when added
            network: this.networkLogs, // Already redacted when added
            state: {
                currentView: this.currentView,
                isAuthenticated: this.isAuthenticated,
                settings: safeSettings
            },
            security: {
                encryption_enabled: this.crypto?.isSupported || false,
                redaction_enabled: !!this.redactor
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `agent-os-debug-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // =========================================================================
    // Settings
    // =========================================================================

    loadSettings() {
        // Load settings into UI
        const debugCheckbox = document.getElementById('setting-debug-mode');
        const verboseCheckbox = document.getElementById('setting-verbose-logging');
        const autoReconnectCheckbox = document.getElementById('setting-auto-reconnect');
        const soundCheckbox = document.getElementById('setting-sound-notifications');
        const darkThemeCheckbox = document.getElementById('setting-dark-theme');
        const ollamaEndpoint = document.getElementById('setting-ollama-endpoint');
        const llamaCppEndpoint = document.getElementById('setting-llama-cpp-endpoint');
        const defaultModel = document.getElementById('setting-default-model');

        if (debugCheckbox) debugCheckbox.checked = this.settings.debug_mode;
        if (verboseCheckbox) verboseCheckbox.checked = this.settings.verbose_logging;
        if (autoReconnectCheckbox) autoReconnectCheckbox.checked = this.settings.auto_reconnect;
        if (soundCheckbox) soundCheckbox.checked = this.settings.sound_notifications;
        if (darkThemeCheckbox) darkThemeCheckbox.checked = this.settings.dark_theme;
        if (ollamaEndpoint) ollamaEndpoint.value = this.settings.ollama_endpoint;
        if (llamaCppEndpoint) llamaCppEndpoint.value = this.settings.llama_cpp_endpoint;
        if (defaultModel) defaultModel.value = this.settings.default_model;
    }

    async updateSetting(key, value) {
        this.settings[key] = value;

        // Convert key format
        const camelKey = key.replace(/_([a-z])/g, (m, c) => c.toUpperCase());

        // Use secure storage for sensitive settings
        const sensitiveKeys = ['ollama_endpoint', 'llama_cpp_endpoint', 'api_key', 'token', 'secret'];
        const isSensitive = sensitiveKeys.some(k => key.toLowerCase().includes(k.replace(/_/g, '')));

        if (isSensitive && this.secureStorage) {
            await this.secureStorage.setItem(camelKey, value);
            // Log without revealing the actual value
            this.addDebugLog('info', `Setting updated: ${key} = [ENCRYPTED]`);
        } else {
            localStorage.setItem(camelKey, value);
            this.addDebugLog('info', `Setting updated: ${key} = ${value}`);
        }

        // Handle specific settings
        if (key === 'dark_theme') {
            // Could toggle light/dark theme here
        }
    }

    // Modal
    setupModal() {
        const modal = document.getElementById('modal');
        const closeBtn = modal.querySelector('.modal-close');

        closeBtn.addEventListener('click', () => this.hideModal());

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideModal();
            }
        });
    }

    showModal(title, content, footer = '') {
        document.getElementById('modal-title').textContent = title;
        document.getElementById('modal-body').innerHTML = content;
        document.getElementById('modal-footer').innerHTML = footer;
        document.getElementById('modal').classList.add('active');
    }

    hideModal() {
        document.getElementById('modal').classList.remove('active');
    }

    // =========================================================================
    // Dreaming Status
    // =========================================================================

    startDreamingPoll() {
        // Initial fetch
        this.fetchDreamingStatus();

        // Poll every 5 seconds (matches backend throttle)
        this.dreamingInterval = setInterval(() => {
            this.fetchDreamingStatus();
        }, 5000);
    }

    async fetchDreamingStatus() {
        try {
            const response = await fetch('/api/system/dreaming');
            if (response.ok) {
                const status = await response.json();
                this.updateDreamingDisplay(status);
            }
        } catch (error) {
            // Silent fail - dreaming is not critical
            if (this.settings.verbose_logging) {
                console.debug('Dreaming status fetch failed:', error.message);
            }
        }
    }

    updateDreamingDisplay(status) {
        const container = document.getElementById('dreaming-status');
        const dot = container?.querySelector('.dreaming-dot');
        const text = container?.querySelector('.dreaming-text');

        if (!container || !dot || !text) return;

        // Update text
        text.textContent = status.message || 'Idle';

        // Update phase class for styling
        container.className = 'dreaming-indicator';
        if (status.phase) {
            container.classList.add(`dreaming-${status.phase}`);
        }

        // Add pulse animation when active
        if (status.phase === 'starting' || status.phase === 'running') {
            dot.classList.add('dreaming-pulse');
        } else {
            dot.classList.remove('dreaming-pulse');
        }

        // Update tooltip with more detail
        const tooltip = `${status.message} (${status.operations_count} operations)`;
        container.setAttribute('data-tooltip', tooltip);
    }

    stopDreamingPoll() {
        if (this.dreamingInterval) {
            clearInterval(this.dreamingInterval);
            this.dreamingInterval = null;
        }
    }

    // Utilities
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showError(message) {
        // Simple alert for now - could be replaced with toast notification
        alert(message);
    }

    loadInitialData() {
        // Load conversations list
        this.loadConversations();
    }

    async loadConversations() {
        try {
            const conversations = await fetch('/api/chat/conversations').then(r => r.json());
            const container = document.getElementById('conversation-list');

            if (conversations.length === 0) {
                container.innerHTML = '<p style="color: var(--text-muted); padding: 0.5rem;">No conversations yet</p>';
                return;
            }

            container.innerHTML = conversations.map(conv => `
                <div class="conversation-item" onclick="app.loadConversation('${conv.id}')">
                    <div style="font-weight: 500;">${this.escapeHtml(conv.title)}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">${conv.message_count} messages</div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }

    async loadConversation(id) {
        this.conversationId = id;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'history' }));
        }
    }
}

// Initialize app
const app = new AgentOS();

// Setup refresh buttons
document.getElementById('refresh-agents-btn')?.addEventListener('click', () => app.loadAgents());
document.getElementById('refresh-system-btn')?.addEventListener('click', () => app.loadSystem());

// Constitution button
document.getElementById('add-rule-btn')?.addEventListener('click', () => app.showAddRuleModal());

// Memory button
document.getElementById('add-memory-btn')?.addEventListener('click', () => app.showAddMemoryModal());

// Contracts button
document.getElementById('create-contract-btn')?.addEventListener('click', () => app.showCreateContractModal());

// Image generation
document.getElementById('image-generation-form')?.addEventListener('submit', (e) => app.generateImage(e));
document.getElementById('refresh-images-btn')?.addEventListener('click', () => app.loadImages());

// Memory search
document.getElementById('memory-search')?.addEventListener('input', async (e) => {
    const query = e.target.value.trim();
    if (query.length < 2) {
        app.loadMemory();
        return;
    }

    try {
        const results = await fetch(`/api/memory/search?query=${encodeURIComponent(query)}`).then(r => r.json());
        const container = document.getElementById('memory-list');
        container.innerHTML = results.map(result => `
            <div class="memory-card">
                <div class="memory-header">
                    <span class="memory-type ${result.entry.memory_type}">${result.entry.memory_type}</span>
                    <span style="color: var(--primary-color);">Score: ${(result.similarity_score * 100).toFixed(0)}%</span>
                </div>
                <div class="memory-content">${app.escapeHtml(result.entry.content)}</div>
                <div class="memory-tags">
                    ${result.entry.tags.map(t => `<span class="tag">${t}</span>`).join('')}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Search failed:', error);
    }
});
