/**
 * DocuRAG Studio — Frontend Application Engine
 * Modular architecture: ThemeManager, ToastManager, ChatController, LibraryController, HistoryController
 */

class ToastManager {
    static show(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `pointer-events-auto px-4 py-3 rounded-xl border text-xs font-medium shadow-xl flex items-center gap-2.5 transition-all duration-300 transform translate-y-2 opacity-0 ${
            type === 'success' ? 'bg-emerald-950/90 text-emerald-300 border-emerald-500/30' :
            type === 'error' ? 'bg-rose-950/90 text-rose-300 border-rose-500/30' :
            'bg-[#0f172a]/95 text-slate-200 border-white/10'
        }`;

        let icon = `<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
        if (type === 'success') {
            icon = `<svg class="w-4 h-4 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`;
        } else if (type === 'error') {
            icon = `<svg class="w-4 h-4 text-rose-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
        }

        toast.innerHTML = `${icon}<span>${message}</span>`;
        container.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => {
            toast.classList.remove('translate-y-2', 'opacity-0');
        });

        setTimeout(() => {
            toast.classList.add('opacity-0', 'translate-y-2');
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }
}

class ThemeManager {
    static init() {
        const toggleBtn = document.getElementById('themeToggle');
        if (!toggleBtn) return;

        const currentTheme = localStorage.getItem('docurag_theme') || 'dark';
        this.applyTheme(currentTheme);

        toggleBtn.addEventListener('click', () => {
            const isDark = document.documentElement.classList.contains('dark');
            const nextTheme = isDark ? 'light' : 'dark';
            this.applyTheme(nextTheme);
        });
    }

    static applyTheme(theme) {
        if (theme === 'light') {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('docurag_theme', 'light');
        } else {
            document.documentElement.classList.add('dark');
            localStorage.setItem('docurag_theme', 'dark');
        }
    }
}

class ModalManager {
    static init() {
        const docsModal = document.getElementById('docsModal');
        const openBtn = document.getElementById('openDocsModalBtn');
        const closeBtn = document.getElementById('closeDocsModalBtn');
        const backdrop = document.getElementById('docsModalBackdrop');

        const openModal = () => {
            if (docsModal) docsModal.classList.remove('hidden');
        };
        const closeModal = () => {
            if (docsModal) docsModal.classList.add('hidden');
        };

        if (openBtn) openBtn.addEventListener('click', openModal);
        if (closeBtn) closeBtn.addEventListener('click', closeModal);
        if (backdrop) backdrop.addEventListener('click', closeModal);

        // Global Escape Key to dismiss open overlays, modals, and drawers
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
                const sourceDrawer = document.getElementById('sourceDrawer');
                if (sourceDrawer) sourceDrawer.classList.add('translate-x-full');

                const sidebar = document.getElementById('sidebar');
                const sidebarBackdrop = document.getElementById('sidebarBackdrop');
                const mobileSidebarBackdrop = document.getElementById('mobileSidebarBackdrop');
                if (sidebar) sidebar.classList.add('-translate-x-full');
                if (sidebarBackdrop) sidebarBackdrop.classList.add('hidden');
                if (mobileSidebarBackdrop) mobileSidebarBackdrop.classList.add('hidden');

                const docDropdown = document.getElementById('docDropdown');
                if (docDropdown) docDropdown.classList.add('hidden');
            }
        });
    }
}

class ChatController {
    constructor() {
        this.cacheDOM();
        if (!this.askBtn && !this.questionInput) return;

        this.currentChatHistory = [];
        this.lastQuery = null;
        this.documents = [];
        this.selectedDocIds = [];

        this.bindEvents();
        this.init();
    }

    cacheDOM() {
        this.askBtn = document.getElementById('askBtn');
        this.questionInput = document.getElementById('questionInput');
        this.chatHistory = document.getElementById('chatHistory');
        this.heroState = document.getElementById('heroState');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        this.exportChatBtn = document.getElementById('exportChatBtn');
        this.chatTitleHeader = document.getElementById('chatTitleHeader');
        this.activeDocTag = document.getElementById('activeDocTag');
        this.chatToolbar = document.getElementById('chatToolbar');

        // Docked input in active chat state
        this.dockedInputContainer = document.getElementById('dockedInputContainer');
        this.dockedQuestionInput = document.getElementById('dockedQuestionInput');
        this.dockedAskBtn = document.getElementById('dockedAskBtn');

        // Document selector
        this.docToggleBtn = document.getElementById('docToggleBtn');
        this.docDropdown = document.getElementById('docDropdown');
        this.docToggleLabel = document.getElementById('docToggleLabel');
        this.documentSelector = document.getElementById('documentSelector');
        this.selectAllDocs = document.getElementById('selectAllDocs');

        // Drawer
        this.sourceDrawer = document.getElementById('sourceDrawer');
        this.drawerTitle = document.getElementById('drawerTitle');
        this.drawerSubtitle = document.getElementById('drawerSubtitle');
        this.drawerContent = document.getElementById('drawerContent');
        this.drawerImageContainer = document.getElementById('drawerImageContainer');
        this.drawerImage = document.getElementById('drawerImage');
        this.closeDrawerBtn = document.getElementById('closeDrawer');

        // Sidebar & Mobile
        this.sidebar = document.getElementById('sidebar');
        this.sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
        this.sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
        this.sidebarBackdrop = document.getElementById('sidebarBackdrop');
        this.mobileMenuToggle = document.getElementById('mobileMenuToggle');
        this.mobileSidebarBackdrop = document.getElementById('mobileSidebarBackdrop');
        this.historySidebarContent = document.getElementById('historySidebarContent');
        this.sidebarDocBadge = document.getElementById('sidebarDocBadge');
    }

    bindEvents() {
        // Main Ask submission
        if (this.askBtn) {
            this.askBtn.addEventListener('click', () => this.handleSend());
        }

        // Textarea auto-resize & keyboard submission
        if (this.questionInput) {
            this.questionInput.addEventListener('input', () => {
                this.questionInput.style.height = 'auto';
                this.questionInput.style.height = Math.min(this.questionInput.scrollHeight, 180) + 'px';
            });

            this.questionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleSend();
                }
            });
        }

        // Docked input listeners
        if (this.dockedAskBtn) {
            this.dockedAskBtn.addEventListener('click', () => {
                if (this.dockedQuestionInput && this.dockedQuestionInput.value.trim()) {
                    this.questionInput.value = this.dockedQuestionInput.value.trim();
                    this.dockedQuestionInput.value = '';
                    this.handleSend();
                }
            });
        }
        if (this.dockedQuestionInput) {
            this.dockedQuestionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (this.dockedQuestionInput.value.trim()) {
                        this.questionInput.value = this.dockedQuestionInput.value.trim();
                        this.dockedQuestionInput.value = '';
                        this.handleSend();
                    }
                }
            });
        }

        // New Chat & Clear Chat
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.resetToNewChat());
        }
        if (this.clearChatBtn) {
            this.clearChatBtn.addEventListener('click', () => this.resetToNewChat());
        }

        // Export Chat
        if (this.exportChatBtn) {
            this.exportChatBtn.addEventListener('click', () => this.exportChatAsMarkdown());
        }

        // Document Dropdown Toggle
        if (this.docToggleBtn && this.docDropdown) {
            this.docToggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.docDropdown.classList.toggle('hidden');
            });
            document.addEventListener('click', (e) => {
                if (!this.docDropdown.contains(e.target) && !this.docToggleBtn.contains(e.target)) {
                    this.docDropdown.classList.add('hidden');
                }
            });
        }

        // Select All Documents Toggle
        if (this.selectAllDocs) {
            this.selectAllDocs.addEventListener('change', (e) => {
                const checkboxes = document.querySelectorAll('.doc-checkbox');
                checkboxes.forEach(cb => cb.checked = e.target.checked);
                this.updateSelectedDocCount();
            });
        }

        // Quick action buttons in Hero state
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const title = btn.querySelector('.text-slate-200')?.innerText || '';
                let prompt = title;
                if (title.toLowerCase().includes('summary')) {
                    prompt = "Summarize the key points, core ideas, and findings of the document in bullet points.";
                } else if (title.toLowerCase().includes('questions')) {
                    prompt = "Extract and formulate 5 test/practice questions based directly on this material, along with their answers.";
                } else if (title.toLowerCase().includes('tables') || title.toLowerCase().includes('metrics')) {
                    prompt = "Extract all significant data tables, numerical metrics, and statistical figures into formatted Markdown tables.";
                }
                this.questionInput.value = prompt;
                this.handleSend();
            });
        });

        // Prompt helper pills
        document.querySelectorAll('.prompt-pill').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.getAttribute('data-prompt') || btn.innerText;
                this.questionInput.value = prompt;
                this.handleSend();
            });
        });

        // Close Drawer
        if (this.closeDrawerBtn) {
            this.closeDrawerBtn.addEventListener('click', () => {
                this.sourceDrawer.classList.add('translate-x-full');
            });
        }

        // Sidebar drawer toggles (Desktop and Mobile)
        const openSidebar = () => {
            if (this.sidebar) this.sidebar.classList.remove('-translate-x-full');
            if (this.sidebarBackdrop) this.sidebarBackdrop.classList.remove('hidden');
        };
        const closeSidebar = () => {
            if (this.sidebar) this.sidebar.classList.add('-translate-x-full');
            if (this.sidebarBackdrop) this.sidebarBackdrop.classList.add('hidden');
        };

        if (this.sidebarToggleBtn) this.sidebarToggleBtn.addEventListener('click', openSidebar);
        if (this.mobileMenuToggle) this.mobileMenuToggle.addEventListener('click', openSidebar);
        if (this.sidebarCloseBtn) this.sidebarCloseBtn.addEventListener('click', closeSidebar);
        if (this.sidebarBackdrop) this.sidebarBackdrop.addEventListener('click', closeSidebar);
        if (this.mobileSidebarBackdrop) this.mobileSidebarBackdrop.addEventListener('click', closeSidebar);
    }

    async init() {
        await this.loadDocuments();
        await this.loadSidebarHistory();

        // Check if navigated with prefilled search or doc params
        const urlParams = new URLSearchParams(window.location.search);
        const queryParam = urlParams.get('q');
        const docParam = urlParams.get('doc');

        if (docParam && this.documents) {
            const targetBox = document.querySelector(`.doc-checkbox[value="${docParam}"]`);
            if (targetBox) {
                const checkboxes = document.querySelectorAll('.doc-checkbox');
                checkboxes.forEach(cb => cb.checked = false);
                targetBox.checked = true;
                if (this.selectAllDocs) this.selectAllDocs.checked = false;
                this.updateSelectedDocCount();
            }
        }

        if (queryParam) {
            this.questionInput.value = queryParam;
            // Clean up the URL to prevent re-query loops on refresh
            if (window.history && window.history.replaceState) {
                const cleanUrl = window.location.pathname + (docParam ? `?doc=${docParam}` : '');
                window.history.replaceState({}, document.title, cleanUrl);
            }
            this.handleSend();
        }
    }

    async apiCall(endpoint, options = {}) {
        const response = await fetch(`/api/${endpoint}`, options);
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || `Server returned ${response.status}`);
        }
        return await response.json();
    }

    async loadDocuments() {
        try {
            const docs = await this.apiCall('documents/');
            this.documents = docs;
            if (this.sidebarDocBadge) {
                this.sidebarDocBadge.innerText = docs.length;
            }

            const loadingText = document.getElementById('docLoadingText');
            if (loadingText) loadingText.style.display = 'none';

            if (!this.documentSelector) return;

            if (docs.length === 0) {
                this.documentSelector.innerHTML = '<p class="text-xs text-slate-500 py-3 text-center">No documents uploaded yet.<br><a href="/library" class="text-cyan-400 underline">Upload in Library</a></p>';
                this.updateSelectedDocCount();
                return;
            }

            this.documentSelector.innerHTML = docs.map(doc => `
                <label class="flex items-center space-x-2.5 cursor-pointer p-2 hover:bg-white/5 rounded-lg transition-colors">
                    <input type="checkbox" value="${doc.id}" class="doc-checkbox form-checkbox h-3.5 w-3.5 text-cyan-500 rounded bg-black/50 border-white/20 focus:ring-cyan-500" checked>
                    <div class="truncate text-xs flex-1">
                        <span class="text-slate-200 block truncate" title="${doc.filename}">${doc.filename}</span>
                        <span class="text-[10px] text-slate-500">${doc.pages || 0} pages • ${doc.status}</span>
                    </div>
                </label>
            `).join('');

            const checkboxes = document.querySelectorAll('.doc-checkbox');
            checkboxes.forEach(cb => cb.addEventListener('change', () => {
                const allChecked = Array.from(checkboxes).every(c => c.checked);
                if (this.selectAllDocs) this.selectAllDocs.checked = allChecked;
                this.updateSelectedDocCount();
            }));

            this.updateSelectedDocCount();
        } catch (e) {
            console.error("Failed loading docs:", e);
            if (this.documentSelector) {
                this.documentSelector.innerHTML = '<p class="text-xs text-rose-400 py-2 text-center">Error loading documents.</p>';
            }
        }
    }

    updateSelectedDocCount() {
        const checkedBoxes = document.querySelectorAll('.doc-checkbox:checked');
        const count = checkedBoxes.length;
        const total = document.querySelectorAll('.doc-checkbox').length;

        this.selectedDocIds = Array.from(checkedBoxes).map(c => c.value);

        if (this.docToggleLabel) {
            if (count === 0) {
                this.docToggleLabel.innerText = "0 Docs";
            } else if (count === total) {
                this.docToggleLabel.innerText = "All Docs";
            } else {
                this.docToggleLabel.innerText = `${count} Docs`;
            }
        }

        if (this.activeDocTag) {
            this.activeDocTag.innerText = count === total ? "All Docs Active" : `${count} Doc(s) Selected`;
        }
    }

    async loadSidebarHistory() {
        if (!this.historySidebarContent) return;
        try {
            const logs = await this.apiCall('history/');
            if (!logs || logs.length === 0) {
                this.historySidebarContent.innerHTML = '<div class="text-center text-slate-500 text-xs py-4">No query logs yet.</div>';
                return;
            }

            this.historySidebarContent.innerHTML = logs.slice(0, 15).map(log => `
                <div class="p-2.5 hover:bg-white/5 rounded-lg transition-colors cursor-pointer group" data-question="${encodeURIComponent(log.question)}">
                    <h4 class="font-medium text-slate-300 text-xs line-clamp-1 group-hover:text-cyan-400 transition-colors">${log.question}</h4>
                    <div class="flex items-center gap-2 mt-1 text-[10px] text-slate-500">
                        <span>⚡ ${log.total_ms || 0}ms</span>
                        <span>•</span>
                        <span>${log.sources ? log.sources.length : 0} sources</span>
                    </div>
                </div>
            `).join('');

            this.historySidebarContent.querySelectorAll('[data-question]').forEach(item => {
                item.addEventListener('click', () => {
                    const q = decodeURIComponent(item.getAttribute('data-question'));
                    this.questionInput.value = q;
                    this.handleSend();
                    if (window.innerWidth < 768 && this.sidebar) {
                        this.sidebar.classList.add('-translate-x-full');
                        this.mobileSidebarBackdrop.classList.add('hidden');
                    }
                });
            });
        } catch (e) {
            console.error("Error loading sidebar history:", e);
        }
    }

    transitionToChatView() {
        if (this.heroState) {
            this.heroState.style.display = 'none';
        }
        if (this.chatToolbar) {
            this.chatToolbar.classList.remove('hidden');
        }
        if (this.chatHistory) {
            this.chatHistory.classList.remove('hidden');
        }
        if (this.dockedInputContainer) {
            this.dockedInputContainer.classList.remove('hidden');
        }
    }

    resetToNewChat() {
        this.currentChatHistory = [];
        this.lastQuery = null;
        if (this.chatHistory) {
            this.chatHistory.innerHTML = '';
            this.chatHistory.classList.add('hidden');
        }
        if (this.dockedInputContainer) {
            this.dockedInputContainer.classList.add('hidden');
        }
        if (this.chatToolbar) {
            this.chatToolbar.classList.add('hidden');
        }
        if (this.heroState) {
            this.heroState.style.display = 'flex';
        }
        if (this.chatTitleHeader) {
            this.chatTitleHeader.innerText = "Studio Session";
        }
        if (this.questionInput) {
            this.questionInput.value = '';
            this.questionInput.style.height = 'auto';
            this.questionInput.focus();
        }
        ToastManager.show("Started a fresh chat session.", "info");
    }

    async handleSend() {
        const question = this.questionInput.value.trim();
        if (!question) return;

        this.lastQuery = question;
        this.transitionToChatView();

        if (this.chatTitleHeader && this.currentChatHistory.length === 0) {
            this.chatTitleHeader.innerText = question.length > 40 ? question.substring(0, 40) + "..." : question;
        }

        // Get selected documents
        const document_ids = this.selectedDocIds;

        // Reset input box
        this.questionInput.value = '';
        this.questionInput.style.height = 'auto';
        this.askBtn.disabled = true;

        // Append user bubble
        this.appendUserMessage(question);

        // Append assistant placeholder with typing animation
        const aiBubbleId = 'msg-' + Date.now();
        const aiBubble = this.appendAIMessage(aiBubbleId);

        try {
            const payload = {
                question,
                document_ids,
                history: this.currentChatHistory
            };

            const result = await this.apiCall('query/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            // Update conversational history memory
            this.currentChatHistory.push({
                user: question,
                assistant: result.answer
            });

            this.updateAIMessage(aiBubble, result);
            this.loadSidebarHistory();
        } catch (err) {
            console.error("Query failed:", err);
            this.updateAIMessage(aiBubble, {
                answer: `⚠️ **Generation Error:** ${err.message}\n\nPlease check your document selection or try again.`
            });
            ToastManager.show(`Error: ${err.message}`, "error");
        } finally {
            this.askBtn.disabled = false;
            this.questionInput.focus();
        }
    }

    appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'flex items-start gap-3.5 flex-row-reverse max-w-4xl mx-auto w-full';
        div.innerHTML = `
            <div class="w-8 h-8 rounded-full bg-cyan-600/30 border border-cyan-400/40 flex items-center justify-center flex-shrink-0 shadow-lg text-cyan-300">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
            </div>
            <div class="bg-[#121927] border border-white/10 text-slate-100 rounded-2xl rounded-tr-sm px-5 py-3.5 max-w-[85%] shadow-xl">
                <p class="whitespace-pre-wrap leading-relaxed text-sm">${this.escapeHTML(text)}</p>
            </div>
        `;
        this.chatHistory.appendChild(div);
        this.scrollToBottom();
    }

    appendAIMessage(id) {
        const div = document.createElement('div');
        div.id = id;
        div.className = 'flex items-start gap-3.5 max-w-4xl mx-auto w-full';
        div.innerHTML = `
            <div class="w-8 h-8 rounded-full bg-cyan-900/40 border border-cyan-500/30 flex items-center justify-center flex-shrink-0 shadow-[0_0_15px_rgba(6,182,212,0.2)] text-cyan-400">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
            <div class="bg-[#0b1019] border border-white/10 rounded-2xl rounded-tl-sm p-5 max-w-[90%] sm:max-w-[85%] w-full shadow-2xl relative">
                <!-- Response Header: Model Metadata Badges -->
                <div class="meta-badges flex items-center gap-2 mb-3 hidden">
                    <span class="latency-badge text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-slate-400 font-mono"></span>
                    <span class="confidence-badge text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-medium"></span>
                    <span class="sources-badge text-[10px] px-2 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 font-medium"></span>
                </div>

                <!-- Animated typing state -->
                <div class="typing-indicator flex items-center space-x-1.5 py-2">
                    <div class="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                    <div class="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style="animation-delay: 0.15s"></div>
                    <div class="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style="animation-delay: 0.3s"></div>
                    <span class="text-xs text-slate-500 pl-2">Retrieving evidence & generating answer...</span>
                </div>

                <!-- Main Response Markdown Content -->
                <div class="message-content markdown-body hidden text-sm"></div>

                <!-- Bottom Sources Drawer Launcher Bar -->
                <div class="sources-container mt-5 pt-3.5 border-t border-white/10 hidden">
                    <div class="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                        <span>Cited Sources</span>
                        <span class="text-[9px] text-cyan-400 font-normal">Click any source or inline citation badge to view page</span>
                    </div>
                    <div class="sources-grid flex flex-wrap gap-2"></div>
                </div>

                <!-- Message Action Bar & Suggested Follow-ups -->
                <div class="action-bar mt-4 pt-3 border-t border-white/5 flex items-center justify-between text-xs text-slate-400 hidden">
                    <div class="flex items-center gap-2">
                        <button class="copy-btn px-2.5 py-1 rounded-md hover:bg-white/10 transition-colors flex items-center gap-1.5 text-slate-400 hover:text-white" title="Copy response">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                            <span>Copy</span>
                        </button>
                        <button class="regen-btn px-2.5 py-1 rounded-md hover:bg-white/10 transition-colors flex items-center gap-1.5 text-slate-400 hover:text-white" title="Regenerate this answer">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                            <span>Regenerate</span>
                        </button>
                    </div>
                </div>

                <!-- Suggested Follow-ups -->
                <div class="followup-suggestions mt-3 flex flex-wrap gap-1.5 hidden"></div>
            </div>
        `;
        this.chatHistory.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    updateAIMessage(div, result) {
        const typing = div.querySelector('.typing-indicator');
        const content = div.querySelector('.message-content');
        const metaBadges = div.querySelector('.meta-badges');
        const latencyBadge = div.querySelector('.latency-badge');
        const confidenceBadge = div.querySelector('.confidence-badge');
        const sourcesBadge = div.querySelector('.sources-badge');
        const sourcesCont = div.querySelector('.sources-container');
        const sourcesGrid = div.querySelector('.sources-grid');
        const actionBar = div.querySelector('.action-bar');
        const copyBtn = div.querySelector('.copy-btn');
        const regenBtn = div.querySelector('.regen-btn');
        const followupContainer = div.querySelector('.followup-suggestions');

        typing.classList.add('hidden');
        content.classList.remove('hidden');
        actionBar.classList.remove('hidden');

        // Metadata badges
        if (result.total_ms || result.confidence || (result.sources && result.sources.length > 0)) {
            metaBadges.classList.remove('hidden');
            if (result.total_ms) {
                latencyBadge.innerText = `⚡ ${result.total_ms}ms`;
            } else {
                latencyBadge.style.display = 'none';
            }

            if (result.confidence) {
                const confPercent = Math.round((result.confidence > 1 ? result.confidence : result.confidence * 100));
                confidenceBadge.innerText = `🎯 Confidence: ${confPercent}%`;
            } else {
                confidenceBadge.style.display = 'none';
            }

            if (result.sources && result.sources.length > 0) {
                sourcesBadge.innerText = `📚 ${result.sources.length} Cited Passage(s)`;
            } else {
                sourcesBadge.style.display = 'none';
            }
        }

        // Parse markdown & inject interactive citation pill spans
        let rawAnswer = result.answer || "No answer generated.";
        let html = typeof marked !== 'undefined' ? marked.parse(rawAnswer) : rawAnswer;

        // Transform [E1] or [1] or [E2, E3] into clickable citation badges
        html = html.replace(/\[(E\d+|\d+)\]/g, (match, citeId) => {
            return `<button class="citation-pill" data-cite="${citeId}"><span>📑 ${citeId}</span></button>`;
        });

        content.innerHTML = html;

        // Add click listener on all citation pills
        content.querySelectorAll('.citation-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                const citeId = pill.getAttribute('data-cite');
                this.findAndOpenCitation(citeId, result.sources);
            });
        });

        // Populate bottom sources grid
        if (result.sources && result.sources.length > 0) {
            sourcesCont.classList.remove('hidden');
            result.sources.forEach(src => {
                const sDiv = document.createElement('button');
                sDiv.className = 'text-xs px-3 py-1.5 bg-white/5 hover:bg-cyan-500/10 border border-white/10 hover:border-cyan-500/30 rounded-xl flex items-center gap-2 cursor-pointer transition-all text-left';
                sDiv.onclick = () => this.openSourceDrawer(src);
                sDiv.innerHTML = `
                    <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                    <span class="font-medium text-cyan-300 truncate max-w-[180px]">${src.document} (p.${src.page})</span>
                `;
                sourcesGrid.appendChild(sDiv);
            });
        }

        // Copy button handler
        if (copyBtn) {
            copyBtn.onclick = async () => {
                try {
                    await navigator.clipboard.writeText(rawAnswer);
                    copyBtn.innerHTML = `<svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg><span class="text-emerald-400">Copied!</span>`;
                    ToastManager.show("Copied response to clipboard!", "success");
                    setTimeout(() => {
                        copyBtn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg><span>Copy</span>`;
                    }, 2500);
                } catch (e) {
                    ToastManager.show("Failed copying to clipboard.", "error");
                }
            };
        }

        // Regenerate button handler
        if (regenBtn && this.lastQuery) {
            regenBtn.onclick = () => {
                this.questionInput.value = this.lastQuery;
                this.handleSend();
            };
        }

        // Add dynamic suggested follow-ups
        if (followupContainer && result.answerable !== false) {
            followupContainer.classList.remove('hidden');
            const followups = [
                "Could you provide more specific examples?",
                "What other sections of the document relate to this?",
                "Can you summarize this in a bulleted table?"
            ];

            followupContainer.innerHTML = followups.map(f => `
                <button class="followup-chip text-[11px] px-2.5 py-1 rounded-full bg-white/5 hover:bg-white/10 border border-white/5 text-slate-300 hover:text-cyan-300 transition-colors">
                    ${f}
                </button>
            `).join('');

            followupContainer.querySelectorAll('.followup-chip').forEach(chip => {
                chip.onclick = () => {
                    this.questionInput.value = chip.innerText.trim();
                    this.handleSend();
                };
            });
        }

        this.scrollToBottom();
    }

    findAndOpenCitation(citeId, sources) {
        if (!sources || sources.length === 0) {
            ToastManager.show(`Citation [${citeId}] details not available.`, "info");
            return;
        }

        // Match by id or index (e.g. E1 -> id="1" or id="E1" or index 0)
        let matched = sources.find(s => String(s.id).toLowerCase() === String(citeId).toLowerCase());
        if (!matched) {
            const num = parseInt(citeId.replace(/\D/g, ''), 10);
            if (!isNaN(num) && num > 0 && num <= sources.length) {
                matched = sources[num - 1];
            }
        }

        if (matched) {
            this.openSourceDrawer(matched);
        } else {
            this.openSourceDrawer(sources[0]);
        }
    }

    openSourceDrawer(src) {
        if (!this.sourceDrawer) return;

        this.drawerTitle.innerText = src.document || "Document Source";
        this.drawerSubtitle.innerText = `Page ${src.page || 1} • Similarity Score: ${Math.round((src.score || 0.9) * 100)}%`;
        this.drawerContent.innerText = src.snippet || "No text snippet available.";

        if (src.thumbnail_url) {
            this.drawerImage.src = src.thumbnail_url;
            this.drawerImageContainer.classList.remove('hidden');
        } else {
            this.drawerImageContainer.classList.add('hidden');
        }

        this.sourceDrawer.classList.remove('translate-x-full');
    }

    exportChatAsMarkdown() {
        if (this.currentChatHistory.length === 0) {
            ToastManager.show("No active chat history to export.", "info");
            return;
        }

        let md = `# DocuRAG Studio — Chat Export\n`;
        md += `*Export Date: ${new Date().toLocaleString()}*\n\n---\n\n`;

        this.currentChatHistory.forEach((turn, idx) => {
            md += `### User\n${turn.user}\n\n`;
            md += `### DocuRAG Assistant\n${turn.assistant}\n\n---\n\n`;
        });

        const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `DocuRAG_Chat_${Date.now()}.md`;
        link.click();
        URL.revokeObjectURL(url);

        ToastManager.show("Exported chat session to Markdown file.", "success");
    }

    escapeHTML(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    scrollToBottom() {
        if (this.chatHistory) {
            this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
        }
    }
}

class LibraryController {
    constructor() {
        this.cacheDOM();
        if (!this.dropZone && !this.docTableBody) return;

        this.allDocuments = [];
        this.bindEvents();
        this.init();
    }

    cacheDOM() {
        this.dropZone = document.getElementById('dropZone');
        this.browseFilesBtn = document.getElementById('browseFilesBtn');
        this.fileInput = document.getElementById('fileInput');
        this.uploadStatus = document.getElementById('uploadStatus');
        this.uploadStatusText = document.getElementById('uploadStatusText');
        this.docTableBody = document.getElementById('docTableBody');
        this.refreshDocsBtn = document.getElementById('refreshDocsBtn');
        this.statTotalDocs = document.getElementById('statTotalDocs');
        this.statReadyDocs = document.getElementById('statReadyDocs');
        this.statTotalChunks = document.getElementById('statTotalChunks');
        this.docSearchInput = document.getElementById('docSearchInput');
        this.docCountBadge = document.getElementById('docCountBadge');
    }

    bindEvents() {
        if (this.dropZone && this.fileInput) {
            // Prevent browser from navigating to dragged files on window
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                window.addEventListener(eventName, (e) => e.preventDefault(), false);
            });

            // Prevent click on hidden input from bubbling
            this.fileInput.addEventListener('click', (e) => {
                e.stopPropagation();
            });

            // Clicking dropZone opens file picker
            this.dropZone.addEventListener('click', (e) => {
                e.preventDefault();
                this.fileInput.value = '';
                this.fileInput.click();
            });

            // Dedicated Browse Files button
            if (this.browseFilesBtn) {
                this.browseFilesBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.fileInput.value = '';
                    this.fileInput.click();
                });
            }

            // Drag and drop events on dropZone
            ['dragenter', 'dragover'].forEach(eventName => {
                this.dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.dropZone.classList.add('drag-over');
                }, false);
            });

            ['dragleave', 'dragend'].forEach(eventName => {
                this.dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    this.dropZone.classList.remove('drag-over');
                }, false);
            });

            this.dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropZone.classList.remove('drag-over');
                const dt = e.dataTransfer;
                if (dt && dt.files && dt.files.length > 0) {
                    this.uploadFiles(Array.from(dt.files));
                }
            }, false);

            this.fileInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.uploadFiles(Array.from(e.target.files));
                }
            });
        }

        if (this.docSearchInput) {
            this.docSearchInput.addEventListener('input', (e) => {
                this.filterDocuments(e.target.value.toLowerCase().trim());
            });
        }

        if (this.refreshDocsBtn) {
            this.refreshDocsBtn.addEventListener('click', () => this.loadDocuments());
        }
    }

    async init() {
        await this.loadDocuments();
    }

    async uploadFiles(files) {
        if (!files || files.length === 0) return;

        const allowed = ['pdf', 'png', 'jpg', 'jpeg', 'webp'];
        const validFiles = [];

        for (const file of files) {
            const ext = file.name.split('.').pop().toLowerCase();
            if (!allowed.includes(ext)) {
                ToastManager.show(`Skipped "${file.name}": Unsupported format. Allowed: PDF, PNG, JPG, JPEG, WEBP`, 'error');
                continue;
            }
            if (file.size > 50 * 1024 * 1024) {
                ToastManager.show(`Skipped "${file.name}": Exceeds 50MB limit.`, 'error');
                continue;
            }
            validFiles.push(file);
        }

        if (validFiles.length === 0) return;

        if (this.uploadStatus) {
            this.uploadStatus.classList.remove('hidden');
        }

        let successCount = 0;
        for (let i = 0; i < validFiles.length; i++) {
            const file = validFiles[i];
            const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
            if (this.uploadStatusText) {
                this.uploadStatusText.innerText = `[${i + 1}/${validFiles.length}] Ingesting "${file.name}" (${sizeMb} MB)...`;
            }

            const formData = new FormData();
            formData.append('file', file);

            const csrfToken = this.getCookie('csrftoken');
            const headers = {};
            if (csrfToken) {
                headers['X-CSRFToken'] = csrfToken;
            }

            try {
                const response = await fetch('/api/documents/upload/', {
                    method: 'POST',
                    headers: headers,
                    body: formData
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(errText || `Server returned ${response.status}`);
                }

                successCount++;
                ToastManager.show(`"${file.name}" uploaded successfully!`, 'success');
            } catch (err) {
                console.error(`Upload failed for ${file.name}:`, err);
                ToastManager.show(`Failed to upload "${file.name}": ${err.message}`, 'error');
            }
        }

        if (this.uploadStatus) {
            this.uploadStatus.classList.add('hidden');
        }
        if (this.fileInput) {
            this.fileInput.value = '';
        }

        if (successCount > 0) {
            await this.loadDocuments();
            this.pollDocumentStatus();
        }
    }

    getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    pollDocumentStatus() {
        const interval = setInterval(async () => {
            const docs = await this.loadDocuments(false);
            const processing = docs.some(d => d.status === 'processing' || d.status === 'queued');
            if (!processing) {
                clearInterval(interval);
            }
        }, 2500);
    }

    async loadDocuments(showFeedback = true) {
        try {
            const response = await fetch('/api/documents/');
            const docs = await response.json();
            this.allDocuments = docs;

            // Update stats
            if (this.statTotalDocs) this.statTotalDocs.innerText = docs.length;
            if (this.statReadyDocs) {
                this.statReadyDocs.innerText = docs.filter(d => d.status === 'ready').length;
            }
            if (this.statTotalChunks) {
                let totalChunks = 0;
                docs.forEach(d => {
                    if (d.chunk_counts) {
                        totalChunks += Object.values(d.chunk_counts).reduce((a, b) => a + b, 0);
                    }
                });
                this.statTotalChunks.innerText = totalChunks;
            }
            if (this.docCountBadge) {
                this.docCountBadge.innerText = `${docs.length} ${docs.length === 1 ? 'File' : 'Files'}`;
            }

            // Filter if text exists in search bar
            const currentFilter = this.docSearchInput ? this.docSearchInput.value.toLowerCase().trim() : '';
            if (currentFilter) {
                this.filterDocuments(currentFilter);
            } else {
                this.renderDocumentTable(docs);
            }

            return docs;
        } catch (e) {
            console.error("Error loading document list:", e);
            if (this.docTableBody) {
                this.docTableBody.innerHTML = `<tr><td colspan="5" class="px-6 py-6 text-center text-rose-400">Failed to load documents: ${e.message}</td></tr>`;
            }
            return [];
        }
    }

    filterDocuments(query) {
        if (!query) {
            this.renderDocumentTable(this.allDocuments);
            return;
        }
        const filtered = this.allDocuments.filter(doc => 
            (doc.filename && doc.filename.toLowerCase().includes(query)) ||
            (doc.id && String(doc.id).toLowerCase().includes(query)) ||
            (doc.status && doc.status.toLowerCase().includes(query)) ||
            (doc.file_type && doc.file_type.toLowerCase().includes(query))
        );
        this.renderDocumentTable(filtered);
    }

    renderDocumentTable(docs) {
        if (!this.docTableBody) return;

        if (docs.length === 0) {
            this.docTableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="px-6 py-10 text-center text-slate-500 dark:text-slate-400">
                        ${this.allDocuments.length === 0 
                            ? 'No documents in your library yet. Drag & drop a PDF or image above to get started!'
                            : 'No documents match your filter criteria.'}
                    </td>
                </tr>
            `;
            return;
        }

        this.docTableBody.innerHTML = docs.map(doc => {
            let statusBadge = '';
            if (doc.status === 'ready') {
                statusBadge = `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-medium text-[11px]"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400"></span>Ready</span>`;
            } else if (doc.status === 'processing' || doc.status === 'queued') {
                statusBadge = `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-medium text-[11px]"><span class="w-2 h-2 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></span>${doc.status}</span>`;
            } else {
                statusBadge = `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-600 dark:text-rose-400 font-medium text-[11px]">Failed</span>`;
            }

            const chunksTotal = doc.chunk_counts ? Object.values(doc.chunk_counts).reduce((a, b) => a + b, 0) : 0;
            const dateStr = doc.uploaded_at ? new Date(doc.uploaded_at).toLocaleDateString() : 'Recent';

            return `
                <tr class="hover:bg-slate-100/60 dark:hover:bg-white/[0.02] transition-colors">
                    <td class="px-6 py-4">
                        <div class="flex items-center gap-3">
                            <div class="w-8 h-8 rounded-lg bg-cyan-500/10 dark:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 flex items-center justify-center font-bold text-xs uppercase">
                                ${doc.file_type || 'PDF'}
                            </div>
                            <div class="truncate max-w-xs sm:max-w-md">
                                <span class="font-semibold text-slate-900 dark:text-white block truncate" title="${this.escapeHTML(doc.filename)}">${this.escapeHTML(doc.filename)}</span>
                                <span class="text-[11px] text-slate-400">ID: ${doc.id}</span>
                            </div>
                        </div>
                    </td>
                    <td class="px-6 py-4 text-slate-600 dark:text-slate-300">
                        <div>${doc.pages || 0} page(s)</div>
                        <div class="text-[11px] text-slate-400">${chunksTotal} chunks</div>
                    </td>
                    <td class="px-6 py-4">${statusBadge}</td>
                    <td class="px-6 py-4 text-slate-500 dark:text-slate-400">${dateStr}</td>
                    <td class="px-6 py-4 text-right">
                        <div class="flex items-center justify-end gap-2">
                            <a href="/ask/?doc=${doc.id}" class="px-2.5 py-1 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 text-xs font-medium transition-colors" title="Chat with this document">Chat</a>
                            <button class="delete-doc-btn px-2.5 py-1 rounded-lg hover:bg-rose-500/10 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 text-xs font-medium transition-colors" data-id="${doc.id}" data-name="${encodeURIComponent(doc.filename)}" title="Delete document">Delete</button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');

        // Bind delete buttons
        this.docTableBody.querySelectorAll('.delete-doc-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.getAttribute('data-id');
                const name = decodeURIComponent(btn.getAttribute('data-name'));
                this.handleDeleteDocument(id, name);
            });
        });
    }

    escapeHTML(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    async handleDeleteDocument(id, name) {
        if (!confirm(`Are you sure you want to delete "${name}"? This removes all indexed chunks and pages.`)) {
            return;
        }

        try {
            const res = await fetch(`/api/documents/${id}/`, { method: 'DELETE' });
            if (!res.ok) throw new Error("Delete failed");
            ToastManager.show(`Deleted "${name}".`, "info");
            await this.loadDocuments();
        } catch (e) {
            ToastManager.show(`Error deleting document: ${e.message}`, "error");
        }
    }
}

class HistoryController {
    constructor() {
        this.cacheDOM();
        if (!this.historyList) return;

        this.allLogs = [];
        this.bindEvents();
        this.init();
    }

    cacheDOM() {
        this.historyList = document.getElementById('historyList');
        this.historySearchInput = document.getElementById('historySearchInput');
        this.historyCountBadge = document.getElementById('historyCountBadge');
    }

    bindEvents() {
        if (this.historySearchInput) {
            this.historySearchInput.addEventListener('input', (e) => {
                this.filterLogs(e.target.value.toLowerCase().trim());
            });
        }
    }

    async init() {
        try {
            const res = await fetch('/api/history/');
            this.allLogs = await res.json();
            this.renderLogs(this.allLogs);
        } catch (e) {
            if (this.historyList) {
                this.historyList.innerHTML = `<div class="text-center py-12 text-rose-400">Error loading logs: ${e.message}</div>`;
            }
        }
    }

    filterLogs(query) {
        if (!query) {
            this.renderLogs(this.allLogs);
            return;
        }
        const filtered = this.allLogs.filter(log =>
            (log.question && log.question.toLowerCase().includes(query)) ||
            (log.answer && log.answer.toLowerCase().includes(query))
        );
        this.renderLogs(filtered);
    }

    renderLogs(logs) {
        if (this.historyCountBadge) {
            this.historyCountBadge.innerText = `${logs.length} ${logs.length === 1 ? 'Query' : 'Queries'}`;
        }

        if (!this.historyList) return;

        if (logs.length === 0) {
            this.historyList.innerHTML = `
                <div class="text-center py-16 bg-white dark:bg-[#0d121c] rounded-2xl border border-slate-200 dark:border-white/10 text-slate-500 dark:text-slate-400">
                    <p class="text-sm">No query logs match your criteria.</p>
                </div>
            `;
            return;
        }

        this.historyList.innerHTML = logs.map((log, index) => {
            const dateStr = log.created_at ? new Date(log.created_at).toLocaleString() : 'Recent';
            const confPercent = Math.round((log.confidence > 1 ? log.confidence : log.confidence * 100));
            const safeAnswer = this.escapeHTML(log.answer || '');
            const safeQuestion = this.escapeHTML(log.question || '');

            return `
                <div class="glass-card rounded-[1.6rem] p-6 shadow-xl space-y-3.5 hover:border-cyan-500/40 dark:hover:border-cyan-400/50 transition-all text-slate-800 dark:text-slate-100">
                    <div class="flex items-start justify-between gap-4">
                        <div class="space-y-1">
                            <span class="text-[11px] text-slate-500 dark:text-slate-400 font-mono">${dateStr}</span>
                            <h3 class="font-bold text-slate-900 dark:text-white text-base">${safeQuestion}</h3>
                        </div>
                        <div class="flex items-center gap-2 flex-shrink-0">
                            <button class="history-copy-btn p-2 rounded-xl bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-white/10 transition-colors" data-index="${index}" title="Copy Answer">
                                <svg class="w-4 h-4 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                            </button>
                            <a href="/ask/?q=${encodeURIComponent(log.question)}" class="px-3.5 py-1.5 rounded-full bg-cyan-600/10 dark:bg-cyan-500/15 hover:bg-cyan-600/20 dark:hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-700 dark:text-cyan-300 text-xs font-semibold transition-all flex items-center gap-1.5 hover:scale-105">
                                <span>Open in Chat</span>
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
                            </a>
                        </div>
                    </div>

                    <div class="history-answer-wrapper relative">
                        <div id="history-answer-${index}" class="history-answer text-xs sm:text-sm text-slate-700 dark:text-slate-300 line-clamp-3 bg-slate-100/70 dark:bg-black/25 p-4 rounded-xl border border-slate-200/80 dark:border-white/5 leading-relaxed font-sans transition-all">
                            ${safeAnswer}
                        </div>
                        <button class="history-expand-btn text-[11px] font-semibold text-cyan-600 dark:text-cyan-400 hover:underline mt-1.5 inline-flex items-center gap-1 cursor-pointer" data-index="${index}">
                            Show more ↓
                        </button>
                    </div>

                    <div class="flex flex-wrap items-center gap-2 pt-1 text-[11px]">
                        <span class="px-2.5 py-0.5 rounded-full bg-slate-200/60 dark:bg-white/5 border border-slate-300/60 dark:border-white/10 text-slate-600 dark:text-slate-400 font-mono">⚡ ${log.total_ms || 0}ms</span>
                        <span class="px-2.5 py-0.5 rounded-full bg-cyan-500/10 dark:bg-cyan-500/15 border border-cyan-500/25 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-300 font-medium">🎯 ${confPercent}% confidence</span>
                        <span class="px-2.5 py-0.5 rounded-full bg-purple-500/10 dark:bg-purple-500/15 border border-purple-500/25 dark:border-purple-500/30 text-purple-700 dark:text-purple-300 font-medium">📚 ${log.sources ? log.sources.length : 0} sources</span>
                    </div>
                </div>
            `;
        }).join('');

        // Wire copy buttons
        this.historyList.querySelectorAll('.history-copy-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.getAttribute('data-index'), 10);
                const log = logs[idx];
                if (log && log.answer) {
                    navigator.clipboard.writeText(log.answer).then(() => {
                        ToastManager.show('Answer copied to clipboard!', 'success');
                    }).catch(() => {
                        ToastManager.show('Failed to copy to clipboard', 'error');
                    });
                }
            });
        });

        // Wire show more / show less toggle
        this.historyList.querySelectorAll('.history-expand-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = btn.getAttribute('data-index');
                const answerEl = document.getElementById(`history-answer-${idx}`);
                if (!answerEl) return;
                const isClamped = answerEl.classList.contains('line-clamp-3');
                if (isClamped) {
                    answerEl.classList.remove('line-clamp-3');
                    btn.innerText = 'Show less ↑';
                } else {
                    answerEl.classList.add('line-clamp-3');
                    btn.innerText = 'Show more ↓';
                }
            });
        });
    }

    escapeHTML(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }
}

// Global App Bootstrapper
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    ModalManager.init();
    window.chatController = new ChatController();
    window.libraryController = new LibraryController();
    window.historyController = new HistoryController();
});
