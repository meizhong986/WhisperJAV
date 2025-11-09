/**
 * WhisperJAV Web GUI - JavaScript Controller
 *
 * Phase 4: Full API Integration with Error Handling
 *
 * Features:
 * - Real PyWebView API integration (no mocks)
 * - Log streaming (100ms polling)
 * - Process status monitoring (500ms polling)
 * - Comprehensive error handling
 * - User-friendly feedback
 * - Tab switching with keyboard navigation
 * - File list management with multi-select
 * - Form validation and state management
 *
 * Last updated: 2025-01-08 (Multi-platform drag-drop support)
 */

// ============================================================
// State Management
// ============================================================
const AppState = {
    // File list
    selectedFiles: [],
    selectedIndices: new Set(),

    // UI state
    activeTab: 'tab1',
    isRunning: false,

    // Process monitoring
    logPollInterval: null,
    statusPollInterval: null,

    // Default output directory
    outputDir: '',

    // Initialization
    async init() {
        await this.loadDefaultOutputDir();
    },

    async loadDefaultOutputDir() {
        try {
            // Call API to get default output directory
            const defaultDir = await pywebview.api.get_default_output_dir();
            this.outputDir = defaultDir;
            document.getElementById('outputDir').value = defaultDir;
        } catch (error) {
            console.error('Failed to load default output directory:', error);
            // Fallback to hardcoded default
            this.outputDir = 'C:\\Users\\Documents\\WhisperJAV\\output';
            document.getElementById('outputDir').value = this.outputDir;
            ConsoleManager.log('Using fallback output directory', 'warning');
        }
    }
};

// ============================================================
// Tab Management
// ============================================================
const TabManager = {
    init() {
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => this.switchTab(button.dataset.tab));
        });

        // Keyboard navigation
        document.querySelector('.tab-bar').addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });
    },

    switchTab(tabId) {
        // Update buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            const isActive = btn.dataset.tab === tabId;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-selected', isActive);
        });

        // Update panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${tabId}-panel`);
        });

        // Update state
        AppState.activeTab = tabId;
    },

    handleKeyboard(e) {
        const tabs = Array.from(document.querySelectorAll('.tab-button'));
        const currentIndex = tabs.findIndex(tab => tab.classList.contains('active'));

        let newIndex = currentIndex;

        if (e.key === 'ArrowLeft') {
            newIndex = Math.max(0, currentIndex - 1);
            e.preventDefault();
        } else if (e.key === 'ArrowRight') {
            newIndex = Math.min(tabs.length - 1, currentIndex + 1);
            e.preventDefault();
        } else if (e.key === 'Home') {
            newIndex = 0;
            e.preventDefault();
        } else if (e.key === 'End') {
            newIndex = tabs.length - 1;
            e.preventDefault();
        } else {
            return;
        }

        if (newIndex !== currentIndex) {
            tabs[newIndex].focus();
            this.switchTab(tabs[newIndex].dataset.tab);
        }
    }
};

// ============================================================
// UI Helpers - Loading States
// ============================================================
const UIHelpers = {
    showLoadingState(button, isLoading) {
        if (isLoading) {
            button.disabled = true;
            button.classList.add('loading');
        } else {
            button.disabled = false;
            button.classList.remove('loading');
        }
    },

    showError(title, message) {
        ErrorHandler.show(title, message);
    },

    showSuccess(title, message) {
        ErrorHandler.showSuccess(title, message);
    }
};

// ============================================================
// File List Management
// ============================================================
const FileListManager = {
    init() {
        const fileList = document.getElementById('fileList');

        // Click handling for selection
        fileList.addEventListener('click', (e) => {
            const item = e.target.closest('.file-item');
            if (item) {
                this.handleItemClick(item, e);
            }
        });

        // Keyboard navigation
        fileList.addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });

        // Drag-and-drop support
        this.initializeDragDrop();

        // Button handlers (real API calls)
        document.getElementById('addFilesBtn').addEventListener('click', () => this.addFiles());
        document.getElementById('addFolderBtn').addEventListener('click', () => this.addFolder());
        document.getElementById('removeSelectedBtn').addEventListener('click', () => this.removeSelected());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearAll());
    },

    initializeDragDrop() {
        const fileList = document.getElementById('fileList');

        fileList.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileList.classList.add('drag-over');
        });

        fileList.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileList.classList.add('drag-over');
        });

        fileList.addEventListener('dragleave', () => {
            fileList.classList.remove('drag-over');
        });

        fileList.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileList.classList.remove('drag-over');

            // Note: Actual file path extraction is handled by Python DOM event handler
            // in main.py using pywebviewFullPath, which bypasses browser security restrictions.
            // This handler just prevents default browser behavior and provides visual feedback.
            // Python will call FileListManager.addDroppedFiles(paths) with the full paths.
        });
    },

    // Method called by Python DOM event handler with full file paths
    addDroppedFiles(paths) {
        if (!Array.isArray(paths) || paths.length === 0) {
            return;
        }

        let addedCount = 0;
        let duplicates = 0;

        paths.forEach(path => {
            if (!AppState.selectedFiles.includes(path)) {
                AppState.selectedFiles.push(path);
                addedCount++;
            } else {
                duplicates++;
            }
        });

        if (addedCount > 0) {
            this.render();
            ConsoleManager.log(`✓ Added ${addedCount} item(s) via drag-and-drop`, 'success');
        }

        if (duplicates > 0) {
            ConsoleManager.log(`ℹ Skipped ${duplicates} duplicate(s)`, 'info');
        }
    },

    render() {
        const fileList = document.getElementById('fileList');
        const emptyState = document.getElementById('emptyState');

        if (AppState.selectedFiles.length === 0) {
            // Show empty state
            emptyState.style.display = 'flex';
            // Remove all file items
            fileList.querySelectorAll('.file-item').forEach(item => item.remove());
            this.updateButtons();
            // Ensure Start button reflects current file presence
            FormManager.validateForm();
            return;
        }

        // Hide empty state
        emptyState.style.display = 'none';

        // Render file items
        const existingItems = Array.from(fileList.querySelectorAll('.file-item'));
        const existingPaths = existingItems.map(item => item.dataset.path);

        AppState.selectedFiles.forEach((file, index) => {
            if (!existingPaths.includes(file)) {
                const item = this.createFileItem(file, index);
                fileList.appendChild(item);
            }
        });

        // Remove items no longer in the list
        existingItems.forEach(item => {
            if (!AppState.selectedFiles.includes(item.dataset.path)) {
                item.remove();
            }
        });

        this.updateButtons();
        // After any change to the file list, re-validate to enable/disable Start
        FormManager.validateForm();
    },

    createFileItem(path, index) {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.dataset.path = path;
        item.dataset.index = index;
        item.tabIndex = 0;

        const icon = document.createElement('span');
        icon.className = 'file-icon';
        // Detect if folder (ends with path separator or no extension)
        const isFolder = path.endsWith('/') || path.endsWith('\\') || !path.includes('.');
        icon.textContent = isFolder ? '📁' : '📄';

        const pathSpan = document.createElement('span');
        pathSpan.className = 'file-path';
        pathSpan.textContent = path;

        item.appendChild(icon);
        item.appendChild(pathSpan);

        return item;
    },

    handleItemClick(item, event) {
        const index = parseInt(item.dataset.index);

        if (event.ctrlKey || event.metaKey) {
            // Ctrl+Click: Toggle selection
            this.toggleSelection(index);
        } else if (event.shiftKey && AppState.selectedIndices.size > 0) {
            // Shift+Click: Range selection
            const lastIndex = Math.max(...Array.from(AppState.selectedIndices));
            this.selectRange(lastIndex, index);
        } else {
            // Normal click: Single selection
            this.selectSingle(index);
        }

        this.updateSelectionUI();
        this.updateButtons();
    },

    toggleSelection(index) {
        if (AppState.selectedIndices.has(index)) {
            AppState.selectedIndices.delete(index);
        } else {
            AppState.selectedIndices.add(index);
        }
    },

    selectSingle(index) {
        AppState.selectedIndices.clear();
        AppState.selectedIndices.add(index);
    },

    selectRange(start, end) {
        const [min, max] = [Math.min(start, end), Math.max(start, end)];
        for (let i = min; i <= max; i++) {
            AppState.selectedIndices.add(i);
        }
    },

    updateSelectionUI() {
        document.querySelectorAll('.file-item').forEach(item => {
            const index = parseInt(item.dataset.index);
            item.classList.toggle('selected', AppState.selectedIndices.has(index));
        });
    },

    updateButtons() {
        const hasFiles = AppState.selectedFiles.length > 0;
        const hasSelection = AppState.selectedIndices.size > 0;

        document.getElementById('removeSelectedBtn').disabled = !hasSelection || AppState.isRunning;
        document.getElementById('clearBtn').disabled = !hasFiles || AppState.isRunning;
    },

    handleKeyboard(e) {
        const items = Array.from(document.querySelectorAll('.file-item'));
        if (items.length === 0) return;

        const currentIndex = Array.from(AppState.selectedIndices).sort((a, b) => b - a)[0] || 0;

        let newIndex = currentIndex;

        if (e.key === 'ArrowDown') {
            newIndex = Math.min(items.length - 1, currentIndex + 1);
            e.preventDefault();
        } else if (e.key === 'ArrowUp') {
            newIndex = Math.max(0, currentIndex - 1);
            e.preventDefault();
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            this.removeSelected();
            e.preventDefault();
            return;
        } else {
            return;
        }

        if (e.shiftKey) {
            this.selectRange(currentIndex, newIndex);
        } else {
            this.selectSingle(newIndex);
        }

        this.updateSelectionUI();
        this.updateButtons();

        // Scroll into view
        if (items[newIndex]) {
            items[newIndex].scrollIntoView({ block: 'nearest' });
        }
    },

    // Real API methods
    async addFiles() {
        const btn = document.getElementById('addFilesBtn');
        UIHelpers.showLoadingState(btn, true);

        try {
            const result = await pywebview.api.select_files();

            if (result.success && result.paths && result.paths.length > 0) {
                // Add selected files (avoid duplicates)
                result.paths.forEach(file => {
                    if (!AppState.selectedFiles.includes(file)) {
                        AppState.selectedFiles.push(file);
                    }
                });

                this.render();
                ConsoleManager.log(`Added ${result.paths.length} file(s)`, 'info');
            }
            // If cancelled, result.success will be false but no error message needed
        } catch (error) {
            ErrorHandler.show('File Selection Error', error.toString());
        } finally {
            UIHelpers.showLoadingState(btn, false);
        }
    },

    async addFolder() {
        const btn = document.getElementById('addFolderBtn');
        UIHelpers.showLoadingState(btn, true);

        try {
            const result = await pywebview.api.select_folder();

            if (result.success && result.path) {
                // Add folder (avoid duplicates)
                if (!AppState.selectedFiles.includes(result.path)) {
                    AppState.selectedFiles.push(result.path);
                }

                this.render();
                ConsoleManager.log(`Added folder: ${result.path}`, 'info');
            }
            // If cancelled, result.success will be false but no error message needed
        } catch (error) {
            ErrorHandler.show('Folder Selection Error', error.toString());
        } finally {
            UIHelpers.showLoadingState(btn, false);
        }
    },

    removeSelected() {
        if (AppState.selectedIndices.size === 0) return;

        // Remove selected items (in reverse order to maintain indices)
        const indicesToRemove = Array.from(AppState.selectedIndices).sort((a, b) => b - a);
        indicesToRemove.forEach(index => {
            AppState.selectedFiles.splice(index, 1);
        });

        AppState.selectedIndices.clear();
        this.render();
        ConsoleManager.log(`Removed ${indicesToRemove.length} item(s)`, 'info');
    },

    clearAll() {
        if (AppState.selectedFiles.length === 0) return;

        const count = AppState.selectedFiles.length;
        AppState.selectedFiles = [];
        AppState.selectedIndices.clear();
        this.render();
        ConsoleManager.log(`Cleared ${count} item(s)`, 'info');
    }
};

// ============================================================
// Form Control Logic
// ============================================================
const FormManager = {
    init() {
        // Model override toggle
        const modelOverrideCheckbox = document.getElementById('modelOverrideEnabled');
        const modelSelect = document.getElementById('modelSelection');

        modelOverrideCheckbox.addEventListener('change', () => {
            modelSelect.disabled = !modelOverrideCheckbox.checked;
        });

        // Validate on form changes
        document.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('change', () => this.validateForm());
        });
    },

    validateForm() {
        const hasFiles = AppState.selectedFiles.length > 0;
        const startBtn = document.getElementById('startBtn');

        // Disable start button if no files selected or process running
        startBtn.disabled = !hasFiles || AppState.isRunning;

        return hasFiles;
    },

    collectFormData() {
        // Collect all form values for API
        const modelOverrideEnabled = document.getElementById('modelOverrideEnabled').checked;
        const asyncProcessingEnabled = document.getElementById('asyncProcessing').checked;

        return {
            // Required fields
            inputs: AppState.selectedFiles,
            output_dir: document.getElementById('outputDir').value,
            mode: document.querySelector('input[name="mode"]:checked').value,
            sensitivity: document.getElementById('sensitivity').value,
            source_language: document.getElementById('source-language').value,
            subs_language: document.getElementById('language').value,

            // Optional verbosity
            verbosity: document.getElementById('verbosity').value,

            // Model override (conditional)
            model_override: modelOverrideEnabled
                ? document.getElementById('modelSelection').value
                : '',

            // Async processing (conditional)
            async_processing: asyncProcessingEnabled,
            max_workers: asyncProcessingEnabled
                ? parseInt(document.getElementById('maxWorkers').value)
                : 1,

            // Other options
            credit: document.getElementById('openingCredit').value.trim(),
            keep_temp: document.getElementById('keepTemp').checked,
            temp_dir: document.getElementById('tempDir').value.trim(),
            accept_cpu_mode: document.getElementById('acceptCpuMode').checked,

            // WIP features (currently disabled, but include for future)
            adaptive_classification: document.getElementById('adaptiveClassification').checked,
            adaptive_audio_enhancement: document.getElementById('adaptiveEnhancement').checked,
            smart_postprocessing: document.getElementById('smartPostprocessing').checked
        };
    }
};

// ============================================================
// Console Management
// ============================================================
const ConsoleManager = {
    init() {
        document.getElementById('clearConsoleBtn').addEventListener('click', () => this.clear());
    },

    log(message, type = 'info') {
        const output = document.getElementById('consoleOutput');
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        line.textContent = message;
        output.appendChild(line);

        // Auto-scroll to bottom (use requestAnimationFrame to ensure DOM updated)
        requestAnimationFrame(() => {
            if (output) {
                output.scrollTop = output.scrollHeight;
            }
        });
    },

    clear() {
        const output = document.getElementById('consoleOutput');
        output.innerHTML = '<div class="console-line">Ready.</div>';
    },

    appendRaw(text) {
        const output = document.getElementById('consoleOutput');

        // Split by lines but preserve empty lines
        const lines = text.split('\n');

        lines.forEach((line, index) => {
            // Skip the last empty line from split
            if (index === lines.length - 1 && line === '') return;

            const lineEl = document.createElement('div');
            lineEl.className = 'console-line';
            lineEl.textContent = line || ' '; // Use space for empty lines to preserve height
            output.appendChild(lineEl);
        });

        // Auto-scroll to bottom (use requestAnimationFrame to ensure DOM updated)
        requestAnimationFrame(() => {
            if (output) {
                output.scrollTop = output.scrollHeight;
            }
        });
    }
};

// ============================================================
// Progress Management
// ============================================================
const ProgressManager = {
    init() {
        this.progressBar = document.getElementById('progressBar');
        this.progressFill = document.getElementById('progressFill');
        this.statusLabel = document.getElementById('statusLabel');
    },

    setIndeterminate(active) {
        this.progressBar.classList.toggle('indeterminate', active);
    },

    setProgress(percent) {
        this.progressBar.classList.remove('indeterminate');
        this.progressFill.style.width = `${percent}%`;
    },

    setStatus(text) {
        this.statusLabel.textContent = text;
    },

    reset() {
        this.setIndeterminate(false);
        this.setProgress(0);
        this.setStatus('Idle');
    }
};

// ============================================================
// Error Handler
// ============================================================
const ErrorHandler = {
    show(title, message) {
        // Log to console
        ConsoleManager.log(`✗ ${title}: ${message}`, 'error');

        // Also show browser alert for critical errors
        // In future, could be replaced with custom modal dialog
        alert(`${title}\n\n${message}`);
    },

    showWarning(title, message) {
        ConsoleManager.log(`⚠ ${title}: ${message}`, 'warning');
    },

    showSuccess(title, message) {
        ConsoleManager.log(`✓ ${title}: ${message}`, 'success');
    }
};

// ============================================================
// Process Management (Real API Integration)
// ============================================================
const ProcessManager = {
    async start() {
        // Validate form
        if (!FormManager.validateForm()) {
            ErrorHandler.show('No Files Selected', 'Please add at least one file or folder before starting.');
            return;
        }

        try {
            // Collect form options
            const options = FormManager.collectFormData();

            // Start process via API
            const result = await pywebview.api.start_process(options);

            if (result.success) {
                // Update UI state
                AppState.isRunning = true;
                this.updateButtonStates();

                // Start progress indication
                ProgressManager.setIndeterminate(true);
                ProgressManager.setStatus('Running...');

                // Log command
                if (result.command) {
                    ConsoleManager.appendRaw(`\n> ${result.command}\n`);
                }

                // Start log polling
                this.startLogPolling();

                // Start status monitoring
                this.startStatusMonitoring();

                ConsoleManager.log('Process started successfully', 'info');
            } else {
                ErrorHandler.show('Start Failed', result.message);
            }
        } catch (error) {
            ErrorHandler.show('Start Error', error.toString());
        }
    },

    async cancel() {
        try {
            const result = await pywebview.api.cancel_process();

            if (result.success) {
                ConsoleManager.log('Process cancelled by user', 'warning');

                // Stop polling
                this.stopLogPolling();
                this.stopStatusMonitoring();

                // Update UI
                AppState.isRunning = false;
                this.updateButtonStates();
                ProgressManager.reset();
                ProgressManager.setStatus('Cancelled');
            } else {
                ErrorHandler.show('Cancel Failed', result.message);
            }
        } catch (error) {
            ErrorHandler.show('Cancel Error', error.toString());
        }
    },

    startLogPolling() {
        // Poll every 100ms for new logs
        AppState.logPollInterval = setInterval(async () => {
            try {
                const logs = await pywebview.api.get_logs();

                if (logs && logs.length > 0) {
                    logs.forEach(line => {
                        // Remove trailing newline if present
                        const cleanLine = line.replace(/\n$/, '');
                        if (cleanLine) {
                            ConsoleManager.appendRaw(cleanLine);
                        }
                    });
                }
            } catch (error) {
                console.error('Log polling error:', error);
                // Don't show error to user - just log to console
            }
        }, 100);
    },

    stopLogPolling() {
        if (AppState.logPollInterval) {
            clearInterval(AppState.logPollInterval);
            AppState.logPollInterval = null;
        }
    },

    startStatusMonitoring() {
        // Poll every 500ms for process status
        AppState.statusPollInterval = setInterval(async () => {
            try {
                const status = await pywebview.api.get_process_status();

                // Update status label
                ProgressManager.setStatus(this.formatStatus(status.status));

                // Check if process finished
                if (status.status === 'completed' ||
                    status.status === 'error' ||
                    status.status === 'cancelled') {

                    // Stop polling
                    this.stopLogPolling();
                    this.stopStatusMonitoring();

                    // Fetch any remaining logs
                    await this.fetchRemainingLogs();

                    // Update UI
                    AppState.isRunning = false;
                    this.updateButtonStates();

                    // Show completion status
                    if (status.status === 'completed') {
                        ProgressManager.setProgress(100);
                        ProgressManager.setStatus('Completed');
                        ErrorHandler.showSuccess('Process Completed', 'Transcription finished successfully');
                    } else if (status.status === 'error') {
                        ProgressManager.reset();
                        ProgressManager.setStatus(`Error (exit code: ${status.exit_code})`);
                        ErrorHandler.show('Process Failed', `Process exited with code ${status.exit_code}. Check console for details.`);
                    } else if (status.status === 'cancelled') {
                        ProgressManager.reset();
                        ProgressManager.setStatus('Cancelled');
                    }
                }
            } catch (error) {
                console.error('Status monitoring error:', error);
                // Don't show error to user - just log to console
            }
        }, 500);
    },

    stopStatusMonitoring() {
        if (AppState.statusPollInterval) {
            clearInterval(AppState.statusPollInterval);
            AppState.statusPollInterval = null;
        }
    },

    async fetchRemainingLogs() {
        // Fetch any remaining logs after process completes
        try {
            const logs = await pywebview.api.get_logs();
            if (logs && logs.length > 0) {
                logs.forEach(line => {
                    const cleanLine = line.replace(/\n$/, '');
                    if (cleanLine) {
                        ConsoleManager.appendRaw(cleanLine);
                    }
                });
            }
        } catch (error) {
            console.error('Error fetching remaining logs:', error);
        }
    },

    formatStatus(status) {
        // Format status for display
        const statusMap = {
            'idle': 'Idle',
            'running': 'Running...',
            'completed': 'Completed',
            'cancelled': 'Cancelled',
            'error': 'Error'
        };
        return statusMap[status] || status;
    },

    updateButtonStates() {
        const startBtn = document.getElementById('startBtn');
        const cancelBtn = document.getElementById('cancelBtn');
        const hasFiles = AppState.selectedFiles.length > 0;

        // Start button: enabled only if files exist and not running
        startBtn.disabled = !hasFiles || AppState.isRunning;

        // Cancel button: enabled only if running
        cancelBtn.disabled = !AppState.isRunning;

        // File management buttons: disabled if running
        FileListManager.updateButtons();
    }
};

// ============================================================
// Directory Controls (Real API Integration)
// ============================================================
const DirectoryControls = {
    init() {
        document.getElementById('browseOutputBtn').addEventListener('click', () => this.browseOutput());
        document.getElementById('openOutputBtn').addEventListener('click', () => this.openOutput());
        document.getElementById('browseTempBtn').addEventListener('click', () => this.browseTemp());
    },

    async browseOutput() {
        try {
            const result = await pywebview.api.select_output_directory();

            if (result.success && result.path) {
                document.getElementById('outputDir').value = result.path;
                AppState.outputDir = result.path;
                ConsoleManager.log(`Output directory: ${result.path}`, 'info');
            }
            // If cancelled, result.success will be false but no error message needed
        } catch (error) {
            ErrorHandler.show('Browse Output Error', error.toString());
        }
    },

    async openOutput() {
        const path = document.getElementById('outputDir').value;

        if (!path) {
            ErrorHandler.showWarning('No Output Directory', 'Please specify an output directory first.');
            return;
        }

        try {
            const result = await pywebview.api.open_output_folder(path);

            if (result.success) {
                ConsoleManager.log(`Opened folder: ${path}`, 'info');
            } else {
                ErrorHandler.show('Open Folder Failed', result.message);
            }
        } catch (error) {
            ErrorHandler.show('Open Folder Error', error.toString());
        }
    },

    async browseTemp() {
        try {
            const result = await pywebview.api.select_folder();

            if (result.success && result.path) {
                document.getElementById('tempDir').value = result.path;
                ConsoleManager.log(`Temp directory: ${result.path}`, 'info');
            }
            // If cancelled, result.success will be false but no error message needed
        } catch (error) {
            ErrorHandler.show('Browse Temp Error', error.toString());
        }
    }
};

// ============================================================
// Run Controls Integration
// ============================================================
const RunControls = {
    init() {
        document.getElementById('startBtn').addEventListener('click', () => ProcessManager.start());
        document.getElementById('cancelBtn').addEventListener('click', () => ProcessManager.cancel());
    }
};

// ============================================================
// About Dialog
// ============================================================
function showAbout() {
    const modal = document.getElementById('aboutModal');
    modal.classList.add('active');

    // Close on overlay click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeAbout();
        }
    });
}

function closeAbout() {
    const modal = document.getElementById('aboutModal');
    modal.classList.remove('active');
}

// ============================================================
// Keyboard Shortcuts
// ============================================================
const KeyboardShortcuts = {
    init() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+O: Add files
            if (e.ctrlKey && e.key === 'o') {
                e.preventDefault();
                FileListManager.addFiles();
            }

            // Ctrl+R: Start processing
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                if (!AppState.isRunning && AppState.selectedFiles.length > 0) {
                    ProcessManager.start();
                }
            }

            // Escape: Cancel process or close modal
            if (e.key === 'Escape') {
                const modal = document.getElementById('aboutModal');
                if (modal.classList.contains('active')) {
                    closeAbout();
                } else if (AppState.isRunning) {
                    ProcessManager.cancel();
                }
            }

            // F1: Show About dialog
            if (e.key === 'F1') {
                e.preventDefault();
                showAbout();
            }

            // F5: Refresh with warning
            if (e.key === 'F5') {
                if (AppState.isRunning) {
                    e.preventDefault();
                    if (confirm('Process is running. Refresh anyway? This will terminate the process.')) {
                        location.reload();
                    }
                }
            }

            // Ctrl+Shift+T: Toggle theme
            if (e.ctrlKey && e.shiftKey && (e.key === 'T' || e.key === 't')) {
                e.preventDefault();
                ThemeManager.toggleTheme();
            }
        });
    }
};

// ============================================================
// Theme Manager (runtime stylesheet switching with persistence)
// ============================================================
const ThemeManager = {
    storageKey: 'wj_theme',
    themes: {
        'default': 'style.css',
        'google': 'style.google.css',
        'carbon': 'style.carbon.css',
        'primer': 'style.primer.css'
    },

    init() {
        // Ensure link element exists
        this.linkEl = document.getElementById('themeStylesheet');
        if (!this.linkEl) {
            console.warn('ThemeManager: #themeStylesheet not found; falling back to style.css');
            return;
        }

        // Apply persisted theme
        const saved = this.getSavedTheme();
        this.applyTheme(saved);

        // Wire UI
        const btn = document.getElementById('themeBtn');
        const menu = document.getElementById('themeMenu');
        if (btn && menu) {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                menu.classList.toggle('active');
                if (menu.classList.contains('active')) {
                    // Focus first option for accessibility
                    const first = menu.querySelector('.theme-option');
                    if (first) first.focus();
                }
            });

            // Close on outside click
            document.addEventListener('click', (e) => {
                if (menu.classList.contains('active')) {
                    menu.classList.remove('active');
                }
            });

            // Close on Escape
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    menu.classList.remove('active');
                }
            });

            // Menu item clicks
            menu.querySelectorAll('.theme-option').forEach(opt => {
                opt.addEventListener('click', (e) => {
                    const theme = opt.dataset.theme;
                    this.applyTheme(theme);
                    menu.classList.remove('active');
                });
            });
        }
    },

    getSavedTheme() {
        try {
            const key = localStorage.getItem(this.storageKey) || 'default';
            return this.themes[key] ? key : 'default';
        } catch (e) {
            return 'default';
        }
    },

    saveTheme(key) {
        try {
            localStorage.setItem(this.storageKey, key);
        } catch (e) {
            // ignore
        }
    },

    applyTheme(key) {
        if (!this.linkEl) return;
        const href = this.themes[key] || this.themes['default'];
        // Apply href and persist
        this.linkEl.setAttribute('href', href);
        this.saveTheme(key in this.themes ? key : 'default');
        ConsoleManager.log(`Theme: ${key}`, 'info');
    },

    toggleTheme() {
        const keys = Object.keys(this.themes);
        const currentHref = this.linkEl ? this.linkEl.getAttribute('href') : 'style.css';
        const currentKey = keys.find(k => this.themes[k] === currentHref) || 'default';
        const idx = keys.indexOf(currentKey);
        const next = keys[(idx + 1) % keys.length];
        this.applyTheme(next);
    }
};

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('WhisperJAV Web GUI - Phase 5 (Production Ready)');

    // Initialize state
    await AppState.init();

    // Initialize components
    TabManager.init();
    FileListManager.init();
    FormManager.init();
    ConsoleManager.init();
    ProgressManager.init();
    DirectoryControls.init();
    RunControls.init();
    KeyboardShortcuts.init();
    ThemeManager.init();

    // Initial validation
    FormManager.validateForm();

    ConsoleManager.log('WhisperJAV GUI initialized', 'success');
    ConsoleManager.log('Ready to process video files', 'info');
    ConsoleManager.log('Press F1 for keyboard shortcuts', 'info');
});

// PyWebView ready event
window.addEventListener('pywebviewready', () => {
    console.log('PyWebView API ready!');
    ConsoleManager.log('PyWebView bridge connected', 'success');

    // Reload default output directory from API
    AppState.loadDefaultOutputDir();
});
