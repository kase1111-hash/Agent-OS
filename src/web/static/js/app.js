/**
 * Agent OS Web Interface
 * Main JavaScript application
 */

class AgentOS {
    constructor() {
        this.ws = null;
        this.conversationId = null;
        this.currentView = 'chat';
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupChat();
        this.setupModal();
        this.connectWebSocket();
        this.loadInitialData();
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
        container.innerHTML = sections.map(section => `
            <div class="section-item" onclick="app.filterRulesBySection('${section.id}')">
                ${section.title}
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
                </div>
                <div class="rule-content">${this.escapeHtml(rule.content)}</div>
                <div class="rule-keywords">
                    ${rule.keywords.map(k => `<span class="keyword">${k}</span>`).join('')}
                </div>
            </div>
        `).join('');
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

// Contracts button
document.getElementById('create-contract-btn')?.addEventListener('click', () => app.showCreateContractModal());

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
