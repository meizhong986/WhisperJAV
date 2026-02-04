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
// Mode-Specific UI Manager (Transformers sensitivity handling)
// ============================================================
const ModeManager = {
    noSensitivityModes: ['transformers'],

    init() {
        const modeSelect = document.getElementById('mode');
        modeSelect.addEventListener('change', (e) => this.handleModeChange(e.target.value));
        // Apply initial state
        this.handleModeChange(modeSelect.value);
    },

    handleModeChange(mode) {
        const sensitivitySelect = document.getElementById('sensitivity');
        const sensitivityLabel = sensitivitySelect.previousElementSibling;
        const transformersInfoRow = document.getElementById('transformersInfoRow');

        if (this.noSensitivityModes.includes(mode)) {
            // Disable sensitivity and show N/A
            sensitivitySelect.disabled = true;
            sensitivitySelect.dataset.previousValue = sensitivitySelect.value;

            // Add N/A option if not exists
            let naOption = sensitivitySelect.querySelector('option[value="n/a"]');
            if (!naOption) {
                naOption = document.createElement('option');
                naOption.value = 'n/a';
                naOption.textContent = 'N/A (not applicable)';
                sensitivitySelect.appendChild(naOption);
            }
            sensitivitySelect.value = 'n/a';

            // Show info row
            if (transformersInfoRow) transformersInfoRow.style.display = 'block';
            sensitivityLabel.textContent = 'Sensitivity (not used for this mode):';
        } else {
            // Re-enable sensitivity
            sensitivitySelect.disabled = false;
            if (sensitivitySelect.dataset.previousValue && sensitivitySelect.dataset.previousValue !== 'n/a') {
                sensitivitySelect.value = sensitivitySelect.dataset.previousValue;
            } else if (sensitivitySelect.value === 'n/a') {
                // Default to aggressive if no previous value
                sensitivitySelect.value = 'aggressive';
            }

            // Hide info row
            if (transformersInfoRow) transformersInfoRow.style.display = 'none';
            sensitivityLabel.textContent = 'Sensitivity (accuracy vs false-positives):';
        }
    },

    isTransformersMode() {
        return document.getElementById('mode').value === 'transformers';
    }
};

// ============================================================
// Transformers Parameter Manager
// ============================================================
const TransformersManager = {
    params: null,
    customized: false,

    defaults: {
        model_id: 'kotoba-tech/kotoba-whisper-bilingual-v1.0',
        chunk_length_s: 15,
        stride_length_s: null,
        batch_size: 8,
        scene: 'none',
        beam_size: 5,
        temperature: 0.0,
        attn_implementation: 'sdpa',
        timestamps: 'segment',
        language: 'ja',
        device: 'auto',
        dtype: 'auto',
    },

    getParams() {
        return this.customized && this.params
            ? { ...this.defaults, ...this.params }
            : { ...this.defaults };
    },

    setParams(params) {
        this.params = params;
        this.customized = true;
    },

    resetToDefaults() {
        this.params = null;
        this.customized = false;
    }
};

// ============================================================
// Qwen3-ASR Manager (v1.8.4+)
// ============================================================
const QwenManager = {
    params: null,
    customized: false,

    defaults: {
        // Model settings
        model_id: 'Qwen/Qwen3-ASR-1.7B',
        device: 'auto',
        dtype: 'auto',
        attn_implementation: 'auto',
        // Processing settings
        batch_size: 1,  // batch_size=1 recommended for accuracy
        max_new_tokens: 4096,
        language: null,  // null = auto-detect
        // Aligner settings
        use_aligner: true,
        aligner_id: 'Qwen/Qwen3-ForcedAligner-0.6B',
        // Post-processing
        japanese_postprocess: true,
        postprocess_preset: 'high_moan',
        // Scene detection (from main dropdown)
        scene: 'semantic',
    },

    getParams() {
        return this.customized && this.params
            ? { ...this.defaults, ...this.params }
            : { ...this.defaults };
    },

    setParams(params) {
        this.params = params;
        this.customized = true;
    },

    resetToDefaults() {
        this.params = null;
        this.customized = false;
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
        const mode = document.getElementById('mode').value;
        const isTransformers = mode === 'transformers';

        // Base options common to all modes
        const baseOptions = {
            inputs: AppState.selectedFiles,
            output_dir: document.getElementById('outputDir').value,
            mode: mode,
            source_language: document.getElementById('source-language').value,
            subs_language: document.getElementById('language').value,
            debug: document.getElementById('debugLogging').checked,
            keep_temp: document.getElementById('keepTemp').checked,
            temp_dir: document.getElementById('tempDir').value.trim(),
            accept_cpu_mode: document.getElementById('acceptCpuMode').checked,
        };

        // Transformers mode: Minimal args by default (use model's internal defaults)
        // Only pass full params if user explicitly customized via Ensemble tab
        if (isTransformers) {
            const isCustomized = TransformersManager.customized;

            // Essential args (always passed)
            const essentialOptions = {
                ...baseOptions,
                hf_device: 'auto',  // Always pass device
                hf_customized: isCustomized,  // Signal to backend
            };

            // If customized, include all HF parameters
            if (isCustomized) {
                const hfParams = TransformersManager.getParams();
                return {
                    ...essentialOptions,
                    hf_model_id: hfParams.model_id,
                    hf_chunk_length: hfParams.chunk_length_s,
                    hf_stride: hfParams.stride_length_s,
                    hf_batch_size: hfParams.batch_size,
                    hf_scene: hfParams.scene,
                    hf_beam_size: hfParams.beam_size,
                    hf_temperature: hfParams.temperature,
                    hf_attn: hfParams.attn_implementation,
                    hf_timestamps: hfParams.timestamps,
                    hf_language: hfParams.language,
                    hf_dtype: hfParams.dtype,
                };
            }

            // Default: minimal args - let model use its tuned defaults
            return essentialOptions;
        }

        // Legacy mode handling (all other modes)
        const modelOverrideEnabled = document.getElementById('modelOverrideEnabled').checked;
        const asyncProcessingEnabled = document.getElementById('asyncProcessing').checked;

        return {
            ...baseOptions,
            sensitivity: document.getElementById('sensitivity').value,

            // Model override (conditional)
            model_override: modelOverrideEnabled
                ? document.getElementById('modelSelection').value
                : '',

            // Async processing (conditional)
            async_processing: asyncProcessingEnabled,

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

            // Capture translation settings before starting (in case user switches tabs)
            TranslateIntegrationManager.captureSettingsOnStart();

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

                        // For Ensemble Mode (tab3), translation is handled by CLI --translate flag
                        // For other modes, trigger separate translation subprocess if enabled
                        const isEnsembleMode = AppState.activeTab === 'tab3';

                        if (isEnsembleMode) {
                            // CLI handled everything including translation (if enabled)
                            ErrorHandler.showSuccess('Process Completed',
                                TranslateIntegrationManager.wasEnabledOnStart()
                                    ? 'Transcription and translation finished successfully'
                                    : 'Transcription finished successfully');
                        } else {
                            // Transcription Mode - translation needs separate subprocess (legacy)
                            ErrorHandler.showSuccess('Process Completed', 'Transcription finished successfully');

                            if (TranslateIntegrationManager.wasEnabledOnStart()) {
                                // Use output_files from API (computed based on mode/language)
                                const outputFiles = status.output_files || [];
                                if (outputFiles.length > 0) {
                                    TranslateIntegrationManager.onTranscriptionComplete(outputFiles);
                                } else {
                                    console.warn('Translation enabled but no output files returned from API');
                                }
                            }
                        }
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
// Two-Pass Ensemble Manager
// ============================================================
const EnsembleManager = {
    // State - Full Configuration Snapshot approach
    state: {
        pass1: {
            pipeline: 'balanced',
            sensitivity: 'aggressive',
            sceneDetector: 'semantic',
            speechEnhancer: 'none',
            speechSegmenter: 'ten',  // TEN VAD default
            model: 'large-v2',
            customized: false,
            params: null,  // null = use defaults, object = full custom config
            isTransformers: false,  // Track if using Transformers pipeline
            isQwen: false,  // Track if using Qwen3-ASR pipeline
            dspEffects: ['loudnorm']  // Default FFmpeg DSP effects
        },
        pass2: {
            enabled: false,
            pipeline: 'fidelity',
            sensitivity: 'aggressive',
            sceneDetector: 'auditok',
            speechEnhancer: 'none',
            speechSegmenter: 'silero',  // Silero v4.0
            model: 'turbo',
            customized: false,
            params: null,
            isTransformers: false,  // Fidelity is a legacy pipeline
            isQwen: false,  // Track if using Qwen3-ASR pipeline
            dspEffects: ['loudnorm']  // Default FFmpeg DSP effects
        },
        mergeStrategy: 'smart_merge',
        currentCustomize: null  // 'pass1' or 'pass2'
    },

    // Model options for different pipeline types
    legacyModels: [
        { value: 'large-v2', label: 'Large V2' },
        { value: 'large-v3', label: 'Large V3' },
        { value: 'turbo', label: 'Turbo' }
    ],
    transformersModels: [
        { value: 'kotoba-tech/kotoba-whisper-bilingual-v1.0', label: 'Kotoba Bilingual v1.0' },
        { value: 'kotoba-tech/kotoba-whisper-v2.2', label: 'Kotoba v2.2 (Latest)' },
        { value: 'kotoba-tech/kotoba-whisper-v2.1', label: 'Kotoba v2.1' },
        { value: 'kotoba-tech/kotoba-whisper-v2.0', label: 'Kotoba v2.0' },
        { value: 'openai/whisper-large-v3-turbo', label: 'Whisper Large v3 Turbo' },
        { value: 'openai/whisper-large-v2', label: 'Whisper Large v2' }
    ],
    qwenModels: [
        { value: 'Qwen/Qwen3-ASR-1.7B', label: 'Qwen3-ASR-1.7B    8GB' },
        { value: 'Qwen/Qwen3-ASR-0.6B', label: 'Qwen3-ASR-0.6B    4GB' }
    ],

    async init() {
        // SYNC: Initialize state from actual HTML form values
        // This handles browser form persistence across page reloads/sessions
        // Without this, checkbox can appear checked but JavaScript state is false

        // Pass 1 state sync
        this.state.pass1.pipeline = document.getElementById('pass1-pipeline').value;
        this.state.pass1.sensitivity = document.getElementById('pass1-sensitivity').value;
        this.state.pass1.sceneDetector = document.getElementById('pass1-scene').value;
        this.state.pass1.speechEnhancer = document.getElementById('pass1-enhancer').value;
        this.state.pass1.speechSegmenter = document.getElementById('pass1-segmenter').value;
        this.state.pass1.model = document.getElementById('pass1-model').value;

        // Pass 2 state sync
        this.state.pass2.enabled = document.getElementById('pass2-enabled').checked;
        this.state.pass2.pipeline = document.getElementById('pass2-pipeline').value;
        this.state.pass2.sensitivity = document.getElementById('pass2-sensitivity').value;
        this.state.pass2.sceneDetector = document.getElementById('pass2-scene').value;
        this.state.pass2.speechEnhancer = document.getElementById('pass2-enhancer').value;
        this.state.pass2.speechSegmenter = document.getElementById('pass2-segmenter').value;
        this.state.pass2.model = document.getElementById('pass2-model').value;

        this.state.mergeStrategy = document.getElementById('merge-strategy').value;

        // Update isTransformers and isQwen flags based on synced pipeline values
        this.state.pass1.isTransformers = this.state.pass1.pipeline === 'transformers';
        this.state.pass1.isQwen = this.state.pass1.pipeline === 'qwen';
        this.state.pass2.isTransformers = this.state.pass2.pipeline === 'transformers';
        this.state.pass2.isQwen = this.state.pass2.pipeline === 'qwen';

        // Pass 2 enable/disable
        document.getElementById('pass2-enabled').addEventListener('change', (e) => {
            this.state.pass2.enabled = e.target.checked;
            this.updatePass2State();
        });

        // Pipeline selection changes with customization warning
        document.getElementById('pass1-pipeline').addEventListener('change', (e) => {
            this.handlePipelineChange('pass1', e.target.value, e.target);
        });
        document.getElementById('pass1-sensitivity').addEventListener('change', (e) => {
            this.handleSensitivityChange('pass1', e.target.value, e.target);
        });

        document.getElementById('pass2-pipeline').addEventListener('change', (e) => {
            this.handlePipelineChange('pass2', e.target.value, e.target);
        });
        document.getElementById('pass2-sensitivity').addEventListener('change', (e) => {
            this.handleSensitivityChange('pass2', e.target.value, e.target);
        });

        // New dropdown event handlers - Pass 1
        document.getElementById('pass1-scene').addEventListener('change', (e) => {
            this.state.pass1.sceneDetector = e.target.value;
        });
        document.getElementById('pass1-enhancer').addEventListener('change', (e) => {
            const selectedOption = e.target.options[e.target.selectedIndex];
            // Block selection of disabled/coming-soon options
            if (selectedOption.disabled || selectedOption.classList.contains('coming-soon')) {
                e.target.value = this.state.pass1.speechEnhancer; // Revert to previous
                return;
            }
            this.state.pass1.speechEnhancer = e.target.value;
            this.updateDspPanel('pass1');
        });
        document.getElementById('pass1-segmenter').addEventListener('change', (e) => {
            this.state.pass1.speechSegmenter = e.target.value;
        });
        document.getElementById('pass1-model').addEventListener('change', (e) => {
            this.state.pass1.model = e.target.value;
        });

        // New dropdown event handlers - Pass 2
        document.getElementById('pass2-scene').addEventListener('change', (e) => {
            this.state.pass2.sceneDetector = e.target.value;
        });
        document.getElementById('pass2-enhancer').addEventListener('change', (e) => {
            const selectedOption = e.target.options[e.target.selectedIndex];
            // Block selection of disabled/coming-soon options
            if (selectedOption.disabled || selectedOption.classList.contains('coming-soon')) {
                e.target.value = this.state.pass2.speechEnhancer; // Revert to previous
                return;
            }
            this.state.pass2.speechEnhancer = e.target.value;
            this.updateDspPanel('pass2');
        });
        document.getElementById('pass2-segmenter').addEventListener('change', (e) => {
            this.state.pass2.speechSegmenter = e.target.value;
        });

        // DSP effects checkbox handlers
        this.initDspCheckboxes();
        document.getElementById('pass2-model').addEventListener('change', (e) => {
            this.state.pass2.model = e.target.value;
        });

        // Merge strategy
        document.getElementById('merge-strategy').addEventListener('change', (e) => {
            this.state.mergeStrategy = e.target.value;
        });

        // Customize buttons
        document.getElementById('customize-pass1').addEventListener('click', () => this.openCustomize('pass1'));
        document.getElementById('customize-pass2').addEventListener('click', () => this.openCustomize('pass2'));

        // Modal controls
        document.getElementById('customizeModalClose').addEventListener('click', () => this.closeModal());
        document.getElementById('customizeModalApply').addEventListener('click', () => this.applyCustomization());
        document.getElementById('customizeModalOK').addEventListener('click', () => this.okAndClose());
        document.getElementById('customizeModalReset').addEventListener('click', () => this.resetToDefaults());

        // Modal tab switching
        document.querySelectorAll('.modal-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchModalTab(tab.dataset.tab));
        });

        // Close modal on overlay click
        document.getElementById('customizeModal').addEventListener('click', (e) => {
            if (e.target.id === 'customizeModal') {
                this.closeModal();
            }
        });

        // Initialize UI state based on synced values
        this.updatePass2State();
        this.updateBadges();
        this.updateRowGreyingState('pass1');
        this.updateRowGreyingState('pass2');
    },

    handlePipelineChange(passKey, newValue, selectElement) {
        const passState = this.state[passKey];
        const isTransformers = newValue === 'transformers';
        const isQwen = newValue === 'qwen';
        const wasTransformers = passState.isTransformers;
        const wasQwen = passState.isQwen;

        // Determine pipeline category for model swapping
        const getPipelineType = (isT, isQ) => isT ? 'transformers' : (isQ ? 'qwen' : 'legacy');
        const oldType = getPipelineType(wasTransformers, wasQwen);
        const newType = getPipelineType(isTransformers, isQwen);

        if (passState.customized) {
            // Warn user that custom params will be reset
            if (confirm(`You have custom parameters for ${passKey === 'pass1' ? 'Pass 1' : 'Pass 2'}. Changing the pipeline will reset them. Continue?`)) {
                passState.pipeline = newValue;
                passState.customized = false;
                passState.params = null;
                passState.isTransformers = isTransformers;
                passState.isQwen = isQwen;
                this.updateBadges();
                this.updateRowGreyingState(passKey);
                // Swap model options if pipeline type changed
                if (oldType !== newType) {
                    this.swapModelOptions(passKey, newType);
                }
            } else {
                // Revert selection
                selectElement.value = passState.pipeline;
            }
        } else {
            passState.pipeline = newValue;
            passState.isTransformers = isTransformers;
            passState.isQwen = isQwen;
            this.updateRowGreyingState(passKey);
            // Swap model options if pipeline type changed
            if (oldType !== newType) {
                this.swapModelOptions(passKey, newType);
            }
        }
    },

    // Swap model dropdown options based on pipeline type
    // pipelineType: 'legacy' | 'transformers' | 'qwen'
    swapModelOptions(passKey, pipelineType) {
        const modelSelect = document.getElementById(`${passKey}-model`);
        let models;

        switch (pipelineType) {
            case 'transformers':
                models = this.transformersModels;
                break;
            case 'qwen':
                models = this.qwenModels;
                break;
            default:
                models = this.legacyModels;
        }

        // Clear existing options
        modelSelect.innerHTML = '';

        // Add new options
        models.forEach((model, index) => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.label;
            if (index === 0) option.selected = true;
            modelSelect.appendChild(option);
        });

        // Update state with first option
        this.state[passKey].model = models[0].value;
    },

    // Update row greying based on pipeline type
    // - Transformers: Disable sensitivity AND segmenter (uses HF internal chunking)
    // - Qwen: Disable sensitivity only, KEEP segmenter (uses post-ASR VAD filter)
    // - Legacy: Enable both
    updateRowGreyingState(passKey) {
        const passState = this.state[passKey];
        const sensitivitySelect = document.getElementById(`${passKey}-sensitivity`);
        const segmenterSelect = document.getElementById(`${passKey}-segmenter`);

        // Check if pass2 is disabled (controls should be disabled regardless of pipeline)
        const isPass2Disabled = passKey === 'pass2' && !this.state.pass2.enabled;

        // Sensitivity: Disabled for both Transformers and Qwen (neither uses sensitivity presets)
        const disableSensitivity = passState.isTransformers || passState.isQwen;

        // Segmenter: Only disabled for Transformers (uses HF internal chunking)
        // Qwen DOES use segmenter as a post-ASR VAD filter (--qwen-segmenter)
        const disableSegmenter = passState.isTransformers;

        if (disableSensitivity) {
            const pipelineName = passState.isTransformers ? 'Transformers' : 'Qwen3-ASR';
            sensitivitySelect.disabled = true;
            sensitivitySelect.title = `Sensitivity not applicable for ${pipelineName} mode`;

            // Set sensitivity to 'none' for visual clarity (if option exists)
            if (sensitivitySelect.querySelector('option[value="none"]')) {
                sensitivitySelect.value = 'none';
                passState.sensitivity = 'none';
            }
        } else {
            // Re-enable sensitivity (unless pass2 is disabled)
            sensitivitySelect.disabled = isPass2Disabled;
            sensitivitySelect.title = '';
        }

        if (disableSegmenter) {
            segmenterSelect.disabled = true;
            segmenterSelect.title = 'Transformers uses HF internal chunking';

            // Set segmenter to 'none' for visual clarity (if option exists)
            if (segmenterSelect.querySelector('option[value="none"]')) {
                segmenterSelect.value = 'none';
                passState.speechSegmenter = 'none';
            }
        } else {
            // Re-enable segmenter (unless pass2 is disabled)
            // Note: Qwen uses segmenter as post-ASR VAD filter
            segmenterSelect.disabled = isPass2Disabled;
            segmenterSelect.title = passState.isQwen ? 'Post-ASR VAD filter for Qwen3-ASR' : '';
        }
    },

    handleSensitivityChange(passKey, newValue, selectElement) {
        const passState = this.state[passKey];

        if (passState.customized) {
            // Warn user that custom params will be reset
            if (confirm(`You have custom parameters for ${passKey === 'pass1' ? 'Pass 1' : 'Pass 2'}. Changing the sensitivity will reset them. Continue?`)) {
                passState.sensitivity = newValue;
                passState.customized = false;
                passState.params = null;
                this.updateBadges();
            } else {
                // Revert selection
                selectElement.value = passState.sensitivity;
            }
        } else {
            passState.sensitivity = newValue;
        }
    },

    updatePass2State() {
        const enabled = this.state.pass2.enabled;
        const pass2Row = document.getElementById('pass2-row');
        const mergeRow = document.getElementById('merge-row');

        // Enable/disable all pass 2 controls
        document.getElementById('pass2-pipeline').disabled = !enabled;
        document.getElementById('pass2-scene').disabled = !enabled;
        document.getElementById('pass2-enhancer').disabled = !enabled;
        document.getElementById('pass2-model').disabled = !enabled;
        document.getElementById('customize-pass2').disabled = !enabled;
        document.getElementById('merge-strategy').disabled = !enabled;

        // Sensitivity and Segmenter handled by updateRowGreyingState (may be additionally disabled for Transformers)
        if (!enabled) {
            document.getElementById('pass2-sensitivity').disabled = true;
            document.getElementById('pass2-segmenter').disabled = true;
        }

        // Visual feedback - grey out entire row
        if (enabled) {
            pass2Row.classList.remove('disabled');
            mergeRow.classList.remove('disabled');
            // Re-apply pipeline-specific greying
            this.updateRowGreyingState('pass2');
        } else {
            pass2Row.classList.add('disabled');
            mergeRow.classList.add('disabled');
        }

        // Update DSP panel visibility for Pass 2
        this.updateDspPanel('pass2');
    },

    // DSP Effects Panel Management
    updateDspPanel(passId) {
        const panel = document.getElementById(`${passId}-dsp-panel`);
        const enhancer = document.getElementById(`${passId}-enhancer`).value;
        const isEnabled = passId === 'pass1' || this.state.pass2.enabled;

        if (panel) {
            // Show panel only when FFmpeg DSP is selected and pass is enabled
            if (enhancer === 'ffmpeg-dsp' && isEnabled) {
                panel.style.display = 'block';
            } else {
                panel.style.display = 'none';
            }
        }
    },

    initDspCheckboxes() {
        // Initialize event listeners for DSP effect checkboxes
        ['pass1', 'pass2'].forEach(passId => {
            const checkboxes = document.querySelectorAll(`input[name="${passId}-dsp"]`);
            checkboxes.forEach(checkbox => {
                checkbox.addEventListener('change', () => {
                    this.updateDspEffectsState(passId);
                });
            });
        });

        // Sync initial state from DOM (in case of browser form persistence)
        this.updateDspEffectsState('pass1');
        this.updateDspEffectsState('pass2');
    },

    updateDspEffectsState(passId) {
        const checkboxes = document.querySelectorAll(`input[name="${passId}-dsp"]:checked`);
        const effects = Array.from(checkboxes).map(cb => cb.value);

        // Update state
        this.state[passId].dspEffects = effects.length > 0 ? effects : ['loudnorm']; // Default to loudnorm if none selected
    },

    getDspEffectsString(passId) {
        // Get comma-separated string of selected DSP effects for CLI
        const effects = this.state[passId].dspEffects;
        return effects.join(',');
    },

    updateBadges() {
        // Update Pass 1 badge
        const pass1Badge = document.getElementById('pass1-badge');
        if (pass1Badge) {
            if (this.state.pass1.customized) {
                pass1Badge.textContent = 'Custom';
                pass1Badge.className = 'pass-badge custom';
            } else {
                pass1Badge.textContent = 'Default';
                pass1Badge.className = 'pass-badge default';
            }
        }

        // Update Pass 2 badge
        const pass2Badge = document.getElementById('pass2-badge');
        if (pass2Badge) {
            if (this.state.pass2.customized) {
                pass2Badge.textContent = 'Custom';
                pass2Badge.className = 'pass-badge custom';
            } else {
                pass2Badge.textContent = 'Default';
                pass2Badge.className = 'pass-badge default';
            }
        }

        // Update button text
        const btn1 = document.getElementById('customize-pass1');
        const btn2 = document.getElementById('customize-pass2');
        if (btn1) {
            btn1.textContent = this.state.pass1.customized ? 'Edit Parameters' : 'Customize Parameters';
        }
        if (btn2) {
            btn2.textContent = this.state.pass2.customized ? 'Edit Parameters' : 'Customize Parameters';
        }
    },

    async loadComponents() {
        // Placeholder for future dynamic loading
    },

    // Parameter categorization for tabs - complete mapping
    paramCategories: {
        transcriber: [
            'temperature', 'compression_ratio_threshold', 'logprob_threshold',
            'logprob_margin', 'no_speech_threshold', 'drop_nonverbal_vocals',
            'condition_on_previous_text', 'initial_prompt', 'word_timestamps',
            'prepend_punctuations', 'append_punctuations', 'clip_timestamps',
            'hallucination_silence_threshold'
        ],
        decoder: [
            'task', 'language', 'best_of', 'beam_size', 'patience',
            'length_penalty', 'prefix', 'suppress_tokens', 'suppress_blank',
            'without_timestamps', 'max_initial_timestamp'
        ],
        vad: [
            'threshold', 'neg_threshold', 'min_speech_duration_ms',
            'max_speech_duration_s', 'min_silence_duration_ms', 'speech_pad_ms'
        ],
        // Internal VAD parameters for kotoba-faster-whisper pipeline
        internalVad: [
            'vad_filter', 'vad_threshold', 'min_speech_duration_ms',
            'max_speech_duration_s', 'min_silence_duration_ms', 'speech_pad_ms'
        ]
        // Engine options = everything else (varies by pipeline backend)
    },

    // Parameter metadata with proper constraints for sliders
    // AUDIT FIX: Ranges aligned with backend Pydantic Field constraints
    parameterMetadata: {
        // === Transcriber parameters ===
        logprob_threshold: { ge: -5.0, le: 0.0, step: 0.1 },           // Fixed: was -1.0, backend allows -5.0
        logprob_margin: { ge: 0.0, le: 5.0, step: 0.1 },               // Fixed: was 1.0, backend allows 5.0
        no_speech_threshold: { ge: 0.0, le: 1.0, step: 0.01 },
        compression_ratio_threshold: { ge: 1.0, le: 5.0, step: 0.1 }, // Fixed: was 0.0-10.0, backend is 1.0-5.0
        temperature: { ge: 0.0, le: 1.0, step: 0.01, isArrayAllowed: true },  // Note: can be array
        hallucination_silence_threshold: { ge: 0.0, le: 10.0, step: 0.1 },    // Fixed: was 5.0, backend allows 10.0

        // === Decoder parameters ===
        patience: { ge: 0.0, le: 5.0, step: 0.1 },
        length_penalty: { ge: 0.0, le: 2.0, step: 0.1 },
        max_initial_timestamp: { ge: 0.0, le: 30.0, step: 0.5 },
        beam_size: { ge: 1, le: 20, step: 1 },                        // Fixed: was 10, backend allows 20
        best_of: { ge: 1, le: 10, step: 1 },

        // === External VAD parameters (Silero VAD for balanced/fidelity) ===
        threshold: { ge: 0.0, le: 1.0, step: 0.01 },
        neg_threshold: { ge: 0.0, le: 1.0, step: 0.01 },

        // === Internal VAD parameters (kotoba-faster-whisper) ===
        // Also applies to external Silero VAD - use larger ranges to cover both
        vad_filter: { type: 'boolean', default: true },
        vad_threshold: { ge: 0.0, le: 1.0, step: 0.01 },
        min_speech_duration_ms: { ge: 0, le: 5000, step: 10 },        // Fixed: was 1000, backend allows 5000
        max_speech_duration_s: { ge: 0.0, le: 300.0, step: 1.0 },     // Fixed: was 60.0, silero allows 300.0
        min_silence_duration_ms: { ge: 0, le: 5000, step: 10 },       // Fixed: was 2000, backend allows 5000
        speech_pad_ms: { ge: 0, le: 2000, step: 10 },                 // Fixed: was 500, backend allows 2000

        // === Engine parameters (NEW - were missing) ===
        repetition_penalty: { ge: 1.0, le: 3.0, step: 0.1 },          // Added: for faster_whisper/kotoba
        no_repeat_ngram_size: { ge: 0, le: 10, step: 1 },             // Added: for faster_whisper/kotoba
        chunk_length: { ge: 1, le: 30, step: 1 },                     // Added: for faster_whisper

        // === Boolean parameters (explicit type for robustness) ===
        suppress_blank: { type: 'boolean', default: true },
        without_timestamps: { type: 'boolean', default: false },
        condition_on_previous_text: { type: 'boolean', default: false },
        word_timestamps: { type: 'boolean', default: true },
        drop_nonverbal_vocals: { type: 'boolean', default: false },
        log_progress: { type: 'boolean', default: false },
        multilingual: { type: 'boolean', default: false },
        regroup: { type: 'boolean', default: true },
        vad: { type: 'boolean', default: true }
    },

    // Model compatibility per pipeline (faster-whisper doesn't support turbo)
    pipelineModelCompatibility: {
        balanced: ['large-v2', 'large-v3'],
        faster: ['large-v2', 'large-v3'],
        fast: ['large-v2', 'large-v3'],
        fidelity: ['turbo', 'large-v2', 'large-v3'],
        'kotoba-faster-whisper': ['kotoba-tech/kotoba-whisper-v2.0-faster', 'RoachLin/kotoba-whisper-v2.2-faster']
    },

    async openCustomize(passKey) {
        // passKey is 'pass1' or 'pass2'
        const passState = this.state[passKey];

        // Route to pipeline-specific handler if applicable
        if (passState.isTransformers) {
            await this.openTransformersCustomize(passKey);
            return;
        }
        if (passState.isQwen) {
            await this.openQwenCustomize(passKey);
            return;
        }

        // Legacy pipeline customize flow
        this.state.currentCustomize = passKey;
        const pipeline = passState.pipeline;
        const sensitivity = passState.sensitivity;

        try {
            // Get resolved pipeline parameters
            const result = await pywebview.api.get_pipeline_defaults(pipeline, sensitivity);

            if (!result.success) {
                ErrorHandler.show('Error', 'Failed to load pipeline parameters: ' + result.error);
                return;
            }

            // Store all params by category
            const allDefaults = {
                transcriber: {},
                decoder: {},
                engine: {},
                vad: {},
                scene: { scene_detection_method: passState.sceneDetector || 'auditok' }
            };

            // Categorize decoder params (from API's decoder section)
            for (const [key, value] of Object.entries(result.params.decoder || {})) {
                if (this.paramCategories.transcriber.includes(key)) {
                    allDefaults.transcriber[key] = value;
                } else if (this.paramCategories.decoder.includes(key)) {
                    allDefaults.decoder[key] = value;
                } else {
                    // Everything else goes to engine
                    allDefaults.engine[key] = value;
                }
            }

            // Categorize provider params (from API's provider section)
            for (const [key, value] of Object.entries(result.params.provider || {})) {
                if (this.paramCategories.transcriber.includes(key)) {
                    allDefaults.transcriber[key] = value;
                } else if (this.paramCategories.decoder.includes(key)) {
                    allDefaults.decoder[key] = value;
                } else if (this.paramCategories.vad.includes(key)) {
                    allDefaults.vad[key] = value;
                } else {
                    // Everything else goes to engine
                    allDefaults.engine[key] = value;
                }
            }

            // VAD params (from API's vad section)
            for (const [key, value] of Object.entries(result.params.vad || {})) {
                if (this.paramCategories.vad.includes(key)) {
                    allDefaults.vad[key] = value;
                } else {
                    // Some VAD sections have engine-specific params (e.g., stable_ts vad/vad_threshold)
                    allDefaults.engine[key] = value;
                }
            }

            // Get scene detection method if available
            if (result.scene_detection_method) {
                allDefaults.scene.scene_detection_method = result.scene_detection_method;
            }

            // Store defaults for reset functionality
            this._currentDefaults = allDefaults;

            // If already customized, use stored params; otherwise use defaults
            let currentValues;
            if (passState.customized && passState.params) {
                currentValues = { ...passState.params };
            } else {
                // Flatten all defaults
                currentValues = {
                    ...allDefaults.transcriber,
                    ...allDefaults.decoder,
                    ...allDefaults.engine,
                    ...allDefaults.vad,
                    ...allDefaults.scene
                };
            }

            // Set modal title
            const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
            const customStatus = passState.customized ? ' [Custom]' : ' [Default]';
            document.getElementById('customizeModalTitle').textContent =
                `${passLabel} Settings (${pipeline} / ${sensitivity})${customStatus}`;

            // Clear all tab panels (new structure: model, quality, segmenter, enhancer, scene)
            const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
            tabs.forEach(tab => {
                const tabEl = document.getElementById(`tab-${tab}`);
                if (tabEl) tabEl.innerHTML = '';
            });

            // Get current model settings
            const currentModel = passState.customized && passState.params.model_name
                ? passState.params.model_name
                : result.model || 'large-v2';
            const currentDevice = passState.customized && passState.params.device
                ? passState.params.device
                : 'cuda';

            // Generate Model tab
            this.generateModelTab('tab-model', currentModel, currentDevice, pipeline);

            // Generate Quality tab (combines transcriber, decoder, and engine params)
            const qualityParams = {
                ...allDefaults.transcriber,
                ...allDefaults.decoder,
                ...allDefaults.engine
            };
            // Determine if this is kotoba pipeline (uses internal VAD)
            const isKotobaPipeline = pipeline === 'kotoba-faster-whisper';
            await this.generateQualityTab('tab-quality', qualityParams, currentValues, isKotobaPipeline, allDefaults.engine);

            // Generate Segmenter tab (backend-specific parameters)
            const segmenterBackend = passState.speechSegmenter || 'silero';
            await this.generateSegmenterTab('tab-segmenter', segmenterBackend, allDefaults.vad, currentValues);

            // Generate Enhancer tab (backend-specific parameters)
            const enhancerBackend = passState.speechEnhancer || 'none';
            await this.generateEnhancerTab('tab-enhancer', enhancerBackend, passState.dspEffects || ['loudnorm']);

            // Generate Scene tab (backend-specific parameters)
            const sceneBackend = passState.sceneDetector || 'auditok';
            await this.generateSceneTab('tab-scene', sceneBackend, currentValues);

            // Reset to first tab
            this.switchModalTab('model');

            // Show modal
            document.getElementById('customizeModal').classList.add('active');

        } catch (error) {
            ErrorHandler.show('Error', 'Failed to open customize dialog: ' + error);
        }
    },

    generateTabControls(tabId, params, currentValues) {
        const container = document.getElementById(tabId);

        if (Object.keys(params).length === 0) {
            container.innerHTML = '<p class="tab-empty">No parameters in this category.</p>';
            return;
        }

        for (const [paramName, defaultValue] of Object.entries(params)) {
            // Look up constraints from metadata
            const metadata = this.parameterMetadata[paramName] || {};
            const param = {
                name: paramName,
                // AUDIT FIX: Pass paramName to inferType for metadata-based type detection
                type: this.inferType(defaultValue, paramName),
                description: '',
                constraints: metadata
            };
            const control = this.generateParamControl(param, currentValues[paramName], defaultValue);
            container.appendChild(control);
        }
    },

    generateSceneDetectionTab(tabId, currentMethod) {
        const container = document.getElementById(tabId);

        const control = document.createElement('div');
        control.className = 'param-control';
        control.dataset.param = 'scene_detection_method';

        const label = document.createElement('label');
        label.textContent = 'Scene Detection Method';
        control.appendChild(label);

        const select = document.createElement('select');
        select.className = 'param-select form-select';

        const options = [
            { value: 'auditok', label: 'Auditok (Energy-based)' },
            { value: 'silero', label: 'Silero (Neural VAD)' },
            { value: 'semantic', label: 'Semantic (Texture Clustering)' }
        ];

        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.value === currentMethod) option.selected = true;
            select.appendChild(option);
        });

        control.appendChild(select);

        const desc = document.createElement('p');
        desc.className = 'param-description';
        desc.textContent = 'Method used to split audio into scenes before transcription.';
        control.appendChild(desc);

        container.appendChild(control);
    },

    generateInternalVadTab(tabId, engineParams, currentValues) {
        const container = document.getElementById(tabId);

        // Internal VAD parameters for kotoba pipeline
        const internalVadParams = {
            vad_filter: engineParams.vad_filter !== undefined ? engineParams.vad_filter : true,
            vad_threshold: engineParams.vad_threshold !== undefined ? engineParams.vad_threshold : 0.01,
            min_speech_duration_ms: engineParams.min_speech_duration_ms || 90,
            max_speech_duration_s: engineParams.max_speech_duration_s || 28.0,
            min_silence_duration_ms: engineParams.min_silence_duration_ms || 150,
            speech_pad_ms: engineParams.speech_pad_ms || 400
        };

        // Add description header
        const descDiv = document.createElement('div');
        descDiv.className = 'tab-description';
        descDiv.innerHTML = `
            <p><strong>Internal VAD (faster-whisper built-in)</strong></p>
            <p>Kotoba pipeline uses faster-whisper's internal Silero VAD for speech detection within each scene.</p>
        `;
        container.appendChild(descDiv);

        // Generate controls for each internal VAD parameter
        for (const [paramName, defaultValue] of Object.entries(internalVadParams)) {
            const currentValue = currentValues[paramName] !== undefined ? currentValues[paramName] : defaultValue;
            const metadata = this.parameterMetadata[paramName] || {};
            const param = {
                name: paramName,
                // AUDIT FIX: Pass paramName to inferType for metadata-based type detection
                type: this.inferType(defaultValue, paramName),
                description: this.getInternalVadDescription(paramName),
                constraints: metadata
            };
            container.innerHTML += this.generateParamControl(param, currentValue);
        }
    },

    getInternalVadDescription(paramName) {
        const descriptions = {
            vad_filter: 'Enable/disable internal VAD filtering (recommended: ON)',
            vad_threshold: 'Speech detection threshold (lower = more sensitive, 0.01 default)',
            min_speech_duration_ms: 'Minimum speech segment duration in milliseconds',
            max_speech_duration_s: 'Maximum speech segment duration before splitting',
            min_silence_duration_ms: 'Minimum silence duration to trigger split',
            speech_pad_ms: 'Padding added around detected speech segments'
        };
        return descriptions[paramName] || '';
    },

    // ========== New Tab Generation Methods (Phase 2) ==========

    /**
     * Generate Quality tab - combines transcriber, decoder, and engine params.
     * For kotoba pipeline, also includes internal VAD params in a separate section.
     */
    async generateQualityTab(tabId, qualityParams, currentValues, isKotobaPipeline, engineParams) {
        const container = document.getElementById(tabId);
        if (!container) return;

        // Group parameters for better organization
        const groups = {
            thresholds: {
                label: 'Quality Thresholds',
                description: 'Control transcription quality and filtering',
                params: ['temperature', 'compression_ratio_threshold', 'logprob_threshold',
                         'logprob_margin', 'no_speech_threshold', 'hallucination_silence_threshold']
            },
            decoding: {
                label: 'Decoding Settings',
                description: 'Control how the model generates text',
                params: ['beam_size', 'best_of', 'patience', 'length_penalty',
                         'suppress_blank', 'without_timestamps', 'condition_on_previous_text']
            },
            advanced: {
                label: 'Advanced Options',
                description: 'Additional parameters for fine-tuning',
                params: ['word_timestamps', 'repetition_penalty', 'no_repeat_ngram_size', 'max_initial_timestamp']
            }
        };

        // Add kotoba internal VAD section if applicable
        if (isKotobaPipeline && engineParams) {
            groups.internalVad = {
                label: 'Internal VAD (Kotoba)',
                description: 'Built-in speech detection for kotoba-faster-whisper',
                params: ['vad_filter', 'vad_threshold', 'min_speech_duration_ms',
                         'max_speech_duration_s', 'min_silence_duration_ms', 'speech_pad_ms']
            };
        }

        // Generate each group
        for (const [groupKey, group] of Object.entries(groups)) {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'param-group';

            // Group header
            const header = document.createElement('div');
            header.className = 'param-group-header';
            header.innerHTML = `<h4>${group.label}</h4><p class="param-group-desc">${group.description}</p>`;
            groupDiv.appendChild(header);

            // Generate controls for params in this group
            let hasParams = false;
            for (const paramName of group.params) {
                // Check if param exists in qualityParams or engineParams
                let defaultValue = qualityParams[paramName];
                if (defaultValue === undefined && engineParams) {
                    defaultValue = engineParams[paramName];
                }
                if (defaultValue === undefined) continue;

                hasParams = true;
                const metadata = this.parameterMetadata[paramName] || {};
                const param = {
                    name: paramName,
                    type: this.inferType(defaultValue, paramName),
                    description: this.getQualityParamDescription(paramName),
                    constraints: metadata
                };
                const currentValue = currentValues[paramName] !== undefined ? currentValues[paramName] : defaultValue;
                const control = this.generateParamControl(param, currentValue, defaultValue);
                groupDiv.appendChild(control);
            }

            if (hasParams) {
                container.appendChild(groupDiv);
            }
        }

        // If no params at all, show empty message
        if (container.children.length === 0) {
            container.innerHTML = '<p class="tab-empty">No quality parameters available for this pipeline.</p>';
        }
    },

    getQualityParamDescription(paramName) {
        const descriptions = {
            temperature: 'Sampling temperature (0 = deterministic, higher = more random). Can be array for fallback.',
            compression_ratio_threshold: 'Skip segments with high compression ratio (repetitive text)',
            logprob_threshold: 'Skip segments with low average log probability',
            logprob_margin: 'Margin for log probability threshold',
            no_speech_threshold: 'Threshold for detecting no speech (higher = stricter)',
            hallucination_silence_threshold: 'Skip silent segments that may cause hallucinations',
            beam_size: 'Number of beams for beam search (higher = better but slower)',
            best_of: 'Number of candidates for sampling',
            patience: 'Patience factor for beam search',
            length_penalty: 'Penalty for long sequences',
            suppress_blank: 'Suppress blank outputs at start of sampling',
            without_timestamps: 'Skip timestamp prediction (faster but no timing)',
            condition_on_previous_text: 'Use previous text as context (can cause loops)',
            word_timestamps: 'Enable word-level timestamps',
            repetition_penalty: 'Penalty for repeating tokens',
            no_repeat_ngram_size: 'Prevent repeating n-grams of this size',
            max_initial_timestamp: 'Maximum initial timestamp position'
        };
        return descriptions[paramName] || '';
    },

    /**
     * Generate Segmenter tab - loads backend-specific parameters from API.
     */
    async generateSegmenterTab(tabId, backend, vadDefaults, currentValues) {
        const container = document.getElementById(tabId);
        if (!container) return;

        try {
            // Load schema from API
            const result = await pywebview.api.get_segmenter_schema(backend);

            if (!result.success) {
                container.innerHTML = `<p class="tab-error">Failed to load segmenter schema: ${result.error}</p>`;
                return;
            }

            // Backend header
            const header = document.createElement('div');
            header.className = 'backend-header';
            header.innerHTML = `
                <h4>${result.display_name || backend}</h4>
                <p class="backend-desc">${result.description || ''}</p>
            `;
            container.appendChild(header);

            // Show info message if present (for "none" backend)
            if (result.info_message) {
                const info = document.createElement('div');
                info.className = 'backend-info';
                info.innerHTML = `<p>${result.info_message}</p>`;
                container.appendChild(info);
            }

            // Generate controls from schema parameters
            const parameters = result.parameters || {};
            const defaults = result.defaults || {};

            if (Object.keys(parameters).length === 0) {
                if (!result.info_message) {
                    container.innerHTML += '<p class="tab-empty">No configurable parameters for this segmenter.</p>';
                }
                return;
            }

            // Group parameters if groups are provided
            const groups = result.groups || { general: Object.keys(parameters) };

            for (const [groupName, paramNames] of Object.entries(groups)) {
                const groupDiv = document.createElement('div');
                groupDiv.className = 'param-group';

                if (groupName !== 'general') {
                    const groupHeader = document.createElement('h5');
                    groupHeader.className = 'param-group-title';
                    groupHeader.textContent = groupName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    groupDiv.appendChild(groupHeader);
                }

                for (const paramName of paramNames) {
                    const paramSchema = parameters[paramName];
                    if (!paramSchema) continue;

                    const defaultValue = defaults[paramName];
                    // Check vadDefaults for legacy values, then currentValues
                    let currentValue = vadDefaults[paramName];
                    if (currentValue === undefined) {
                        currentValue = currentValues[paramName];
                    }
                    if (currentValue === undefined) {
                        currentValue = defaultValue;
                    }

                    const control = this.generateSchemaControl(paramName, paramSchema, currentValue, defaultValue);
                    groupDiv.appendChild(control);
                }

                container.appendChild(groupDiv);
            }

        } catch (error) {
            container.innerHTML = `<p class="tab-error">Error loading segmenter parameters: ${error}</p>`;
        }
    },

    /**
     * Generate Enhancer tab - loads backend-specific parameters from API.
     */
    async generateEnhancerTab(tabId, backend, currentDspEffects) {
        const container = document.getElementById(tabId);
        if (!container) return;

        try {
            // Load schema from API
            const result = await pywebview.api.get_enhancer_schema(backend);

            if (!result.success) {
                container.innerHTML = `<p class="tab-error">Failed to load enhancer schema: ${result.error}</p>`;
                return;
            }

            // Backend header
            const header = document.createElement('div');
            header.className = 'backend-header';
            header.innerHTML = `
                <h4>${result.display_name || backend}</h4>
                <p class="backend-desc">${result.description || ''}</p>
            `;
            container.appendChild(header);

            // Show info message if present
            if (result.info_message) {
                const info = document.createElement('div');
                info.className = 'backend-info';
                info.innerHTML = `<p>${result.info_message}</p>`;
                container.appendChild(info);
            }

            // Generate controls from schema parameters
            const parameters = result.parameters || {};
            const defaults = result.defaults || {};

            if (Object.keys(parameters).length === 0) {
                if (!result.info_message) {
                    container.innerHTML += '<p class="tab-empty">No configurable parameters for this enhancer.</p>';
                }
                return;
            }

            // Special handling for FFmpeg DSP effects checkbox list
            for (const [paramName, paramSchema] of Object.entries(parameters)) {
                if (paramSchema.type === 'checkbox_list') {
                    const control = this.generateCheckboxListControl(
                        paramName,
                        paramSchema,
                        backend === 'ffmpeg-dsp' ? currentDspEffects : paramSchema.default
                    );
                    container.appendChild(control);
                } else {
                    const defaultValue = defaults[paramName];
                    const control = this.generateSchemaControl(paramName, paramSchema, defaultValue, defaultValue);
                    container.appendChild(control);
                }
            }

        } catch (error) {
            container.innerHTML = `<p class="tab-error">Error loading enhancer parameters: ${error}</p>`;
        }
    },

    /**
     * Generate checkbox list control for FFmpeg DSP effects.
     */
    generateCheckboxListControl(paramName, schema, currentValues) {
        const container = document.createElement('div');
        container.className = 'param-control checkbox-list-control';
        container.dataset.param = paramName;

        const label = document.createElement('label');
        label.textContent = schema.label || paramName;
        container.appendChild(label);

        if (schema.description) {
            const desc = document.createElement('p');
            desc.className = 'param-description';
            desc.textContent = schema.description;
            container.appendChild(desc);
        }

        const checkboxContainer = document.createElement('div');
        checkboxContainer.className = 'checkbox-list';

        const options = schema.options || [];
        const selectedValues = Array.isArray(currentValues) ? currentValues : (schema.default || []);

        for (const opt of options) {
            const checkboxWrapper = document.createElement('div');
            checkboxWrapper.className = 'checkbox-item';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `${paramName}-${opt.value}`;
            checkbox.value = opt.value;
            checkbox.checked = selectedValues.includes(opt.value);
            checkbox.className = 'effect-checkbox';

            const checkboxLabel = document.createElement('label');
            checkboxLabel.htmlFor = checkbox.id;
            checkboxLabel.className = 'checkbox-label';

            let labelHtml = `<span class="effect-name">${opt.label}</span>`;
            if (opt.description) {
                labelHtml += `<span class="effect-desc">${opt.description}</span>`;
            }
            if (opt.warning) {
                labelHtml += `<span class="effect-warning">${opt.warning}</span>`;
            }
            checkboxLabel.innerHTML = labelHtml;

            checkboxWrapper.appendChild(checkbox);
            checkboxWrapper.appendChild(checkboxLabel);
            checkboxContainer.appendChild(checkboxWrapper);
        }

        container.appendChild(checkboxContainer);
        return container;
    },

    /**
     * Generate Scene tab - loads backend-specific parameters from API.
     */
    async generateSceneTab(tabId, backend, currentValues) {
        const container = document.getElementById(tabId);
        if (!container) return;

        try {
            // Backend selector
            const selectorDiv = document.createElement('div');
            selectorDiv.className = 'param-control';
            selectorDiv.dataset.param = 'scene_detection_method';

            const selectorLabel = document.createElement('label');
            selectorLabel.textContent = 'Scene Detection Method';
            selectorDiv.appendChild(selectorLabel);

            const select = document.createElement('select');
            select.className = 'param-select form-select';
            select.id = 'scene-method-select';

            const methodOptions = [
                { value: 'auditok', label: 'Auditok (Energy-based)' },
                { value: 'silero', label: 'Silero (Neural VAD)' },
                { value: 'semantic', label: 'Semantic (Texture Clustering)' },
                { value: 'none', label: 'None (Skip Scene Detection)' }
            ];

            methodOptions.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                if (opt.value === backend) option.selected = true;
                select.appendChild(option);
            });

            selectorDiv.appendChild(select);
            container.appendChild(selectorDiv);

            // Load and show backend-specific parameters
            const result = await pywebview.api.get_scene_detector_schema(backend);

            if (!result.success) {
                const error = document.createElement('p');
                error.className = 'tab-error';
                error.textContent = `Failed to load scene detector schema: ${result.error}`;
                container.appendChild(error);
                return;
            }

            // Description
            if (result.description) {
                const desc = document.createElement('p');
                desc.className = 'backend-desc';
                desc.textContent = result.description;
                container.appendChild(desc);
            }

            // Info message for "none" backend
            if (result.info_message) {
                const info = document.createElement('div');
                info.className = 'backend-info';
                info.innerHTML = `<p>${result.info_message}</p>`;
                container.appendChild(info);
            }

            // Parameters
            const parameters = result.parameters || {};
            const defaults = result.defaults || {};

            if (Object.keys(parameters).length > 0) {
                const paramsDiv = document.createElement('div');
                paramsDiv.className = 'scene-params';
                paramsDiv.id = 'scene-params-container';

                for (const [paramName, paramSchema] of Object.entries(parameters)) {
                    const defaultValue = defaults[paramName];
                    const currentValue = currentValues[paramName] !== undefined
                        ? currentValues[paramName]
                        : defaultValue;

                    const control = this.generateSchemaControl(paramName, paramSchema, currentValue, defaultValue);
                    paramsDiv.appendChild(control);
                }

                container.appendChild(paramsDiv);
            }

        } catch (error) {
            container.innerHTML = `<p class="tab-error">Error loading scene detector parameters: ${error}</p>`;
        }
    },

    /**
     * Generate a control from API schema format.
     */
    generateSchemaControl(paramName, schema, currentValue, defaultValue) {
        const container = document.createElement('div');
        container.className = 'param-control';
        container.dataset.param = paramName;

        const label = document.createElement('label');
        label.textContent = schema.label || paramName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        container.appendChild(label);

        let input;
        const type = schema.type || 'text';

        switch (type) {
            case 'slider':
                input = document.createElement('input');
                input.type = 'range';
                input.className = 'param-slider';
                input.min = schema.min || 0;
                input.max = schema.max || 1;
                input.step = schema.step || 0.01;
                input.value = currentValue !== undefined ? currentValue : defaultValue;

                // Value display
                const valueDisplay = document.createElement('span');
                valueDisplay.className = 'slider-value';
                valueDisplay.textContent = input.value;
                input.addEventListener('input', () => {
                    valueDisplay.textContent = input.value;
                });

                const sliderWrapper = document.createElement('div');
                sliderWrapper.className = 'slider-wrapper';
                sliderWrapper.appendChild(input);
                sliderWrapper.appendChild(valueDisplay);
                container.appendChild(sliderWrapper);
                break;

            case 'dropdown':
                input = document.createElement('select');
                input.className = 'param-select form-select';

                const options = schema.options || [];
                options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt.value;
                    option.textContent = opt.label;
                    if (opt.value === currentValue || opt.value === defaultValue) {
                        option.selected = true;
                    }
                    input.appendChild(option);
                });
                container.appendChild(input);
                break;

            case 'spinner':
            case 'number':
                input = document.createElement('input');
                input.type = 'number';
                input.className = 'param-spinner';
                input.min = schema.min || 0;
                input.max = schema.max || 100;
                input.step = schema.step || 1;
                input.value = currentValue !== undefined ? currentValue : defaultValue;
                container.appendChild(input);
                break;

            case 'toggle':
            case 'boolean':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.className = 'param-checkbox';
                input.checked = currentValue !== undefined ? currentValue : defaultValue;
                container.appendChild(input);
                break;

            default:
                input = document.createElement('input');
                input.type = 'text';
                input.className = 'param-text';
                input.value = currentValue !== undefined ? currentValue : (defaultValue || '');
                container.appendChild(input);
        }

        // Description
        if (schema.description) {
            const desc = document.createElement('p');
            desc.className = 'param-description';
            desc.textContent = schema.description;
            container.appendChild(desc);
        }

        return container;
    },

    // ========== Transformers Pipeline Customization ==========

    async openTransformersCustomize(passKey) {
        this.state.currentCustomize = passKey;
        const passState = this.state[passKey];

        try {
            // Get Transformers parameter schema from API
            const result = await pywebview.api.get_transformers_schema();

            if (!result.success) {
                ErrorHandler.show('Error', 'Failed to load Transformers parameters: ' + (result.error || 'Unknown error'));
                return;
            }

            // Set modal title
            const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
            const customStatus = passState.customized ? ' [Custom]' : ' [Default]';
            document.getElementById('customizeModalTitle').textContent =
                `${passLabel} Settings (Transformers / HuggingFace)${customStatus}`;

            // Store schema for reset functionality
            this._transformersSchema = result.schema;

            // Get current values (use stored params or defaults)
            const currentValues = passState.customized && passState.params
                ? { ...passState.params }
                : { ...TransformersManager.defaults };

            // Generate Transformers-specific tabs
            this.generateTransformersTabs(result.schema, currentValues);

            // Show modal
            document.getElementById('customizeModal').classList.add('active');

        } catch (error) {
            ErrorHandler.show('Error', 'Failed to open Transformers customize dialog: ' + error);
        }
    },

    generateTransformersTabs(schema, currentValues) {
        // Update tab labels for Transformers context
        // New structure: model, quality, segmenter, enhancer, scene
        const tabLabels = {
            'model': 'Model',
            'quality': 'Quality',
            'segmenter': 'Chunking',
            'enhancer': 'Enhancer',
            'scene': 'Scene'
        };

        // Clear all tab panels and update labels
        const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
        tabs.forEach(tab => {
            const panel = document.getElementById(`tab-${tab}`);
            if (panel) panel.innerHTML = '';
            const tabBtn = document.querySelector(`[data-tab="${tab}"]`);
            if (tabBtn) {
                tabBtn.textContent = tabLabels[tab];
                // Show all tabs for Transformers
                tabBtn.style.display = '';
            }
        });

        // Generate Model tab
        this.generateTransformersModelTab('tab-model', schema.model, currentValues);

        // Generate Quality tab (beam_size, temperature, etc.)
        this.generateTransformersQualityTab('tab-quality', schema.quality, currentValues);

        // Generate Chunking tab (in Segmenter slot - chunk_length, stride, batch_size)
        this.generateTransformersChunkingTab('tab-segmenter', schema.chunking, currentValues);

        // Enhancer tab - not used for Transformers (speech enhancement happens before)
        document.getElementById('tab-enhancer').innerHTML =
            '<p class="tab-empty">Speech enhancement is configured in the main Ensemble tab.<br>Transformers mode uses audio as-is from the enhancement stage.</p>';

        // Generate Scene tab
        this.generateTransformersSceneTab('tab-scene', schema.scene, currentValues);

        // Reset to first tab
        this.switchModalTab('model');
    },

    generateTransformersModelTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Model ID dropdown
        const modelDef = schemaSection.model_id;
        container.appendChild(this.createTransformersDropdown(
            'model_id', 'Model',
            modelDef.options,
            currentValues.model_id || modelDef.default,
            'HuggingFace model identifier'
        ));

        // Device dropdown
        const deviceDef = schemaSection.device;
        container.appendChild(this.createTransformersDropdown(
            'device', 'Device',
            deviceDef.options,
            currentValues.device || deviceDef.default,
            'Compute device (auto will detect GPU availability)'
        ));

        // Data type dropdown
        const dtypeDef = schemaSection.dtype;
        container.appendChild(this.createTransformersDropdown(
            'dtype', 'Data Type',
            dtypeDef.options,
            currentValues.dtype || dtypeDef.default,
            'Model precision (auto selects based on hardware)'
        ));
    },

    generateTransformersChunkingTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Chunk length slider
        const chunkDef = schemaSection.chunk_length_s;
        container.appendChild(this.createTransformersSlider(
            'chunk_length_s', chunkDef.label,
            chunkDef.min, chunkDef.max, chunkDef.step,
            currentValues.chunk_length_s ?? chunkDef.default,
            'Length of audio chunks for parallel processing'
        ));

        // Stride slider
        const strideDef = schemaSection.stride_length_s;
        container.appendChild(this.createTransformersSlider(
            'stride_length_s', strideDef.label,
            strideDef.min, strideDef.max, strideDef.step,
            currentValues.stride_length_s ?? (currentValues.chunk_length_s ? Math.floor(currentValues.chunk_length_s / 6) : 2),
            'Overlap between chunks (null = auto-calculate as chunk/6)'
        ));

        // Batch size slider
        const batchDef = schemaSection.batch_size;
        container.appendChild(this.createTransformersSlider(
            'batch_size', batchDef.label,
            batchDef.min, batchDef.max, batchDef.step,
            currentValues.batch_size ?? batchDef.default,
            'Number of chunks to process simultaneously (reduce if OOM)'
        ));
    },

    generateTransformersQualityTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Beam size slider
        const beamDef = schemaSection.beam_size;
        container.appendChild(this.createTransformersSlider(
            'beam_size', beamDef.label,
            beamDef.min, beamDef.max, beamDef.step,
            currentValues.beam_size ?? beamDef.default,
            'Beam search width (higher = better quality, slower)'
        ));

        // Temperature slider
        const tempDef = schemaSection.temperature;
        container.appendChild(this.createTransformersSlider(
            'temperature', tempDef.label,
            tempDef.min, tempDef.max, tempDef.step,
            currentValues.temperature ?? tempDef.default,
            'Sampling temperature (0 = greedy decoding)'
        ));

        // Attention implementation dropdown
        const attnDef = schemaSection.attn_implementation;
        container.appendChild(this.createTransformersDropdown(
            'attn_implementation', attnDef.label,
            attnDef.options,
            currentValues.attn_implementation || attnDef.default,
            'Attention mechanism (SDPA is fastest for most GPUs)'
        ));

        // Timestamps dropdown
        const tsDef = schemaSection.timestamps;
        container.appendChild(this.createTransformersDropdown(
            'timestamps', tsDef.label,
            tsDef.options,
            currentValues.timestamps || tsDef.default,
            'Timestamp granularity (word requires batch_size=1)'
        ));
    },

    generateTransformersSceneTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Scene detection dropdown
        const sceneDef = schemaSection.scene;
        container.appendChild(this.createTransformersDropdown(
            'scene', sceneDef.label,
            sceneDef.options,
            currentValues.scene || sceneDef.default,
            'Split audio into scenes before processing (none = process whole file)'
        ));
    },

    createTransformersDropdown(paramName, label, options, currentValue, description) {
        const control = document.createElement('div');
        control.className = 'param-control';
        control.dataset.param = paramName;

        const labelEl = document.createElement('label');
        labelEl.textContent = label;
        control.appendChild(labelEl);

        const select = document.createElement('select');
        select.className = 'param-select form-select';
        select.id = `hf-${paramName}`;

        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.value === currentValue) option.selected = true;
            select.appendChild(option);
        });

        control.appendChild(select);

        if (description) {
            const desc = document.createElement('p');
            desc.className = 'param-description';
            desc.textContent = description;
            control.appendChild(desc);
        }

        return control;
    },

    createTransformersSlider(paramName, label, min, max, step, currentValue, description) {
        const control = document.createElement('div');
        control.className = 'param-control';
        control.dataset.param = paramName;

        const labelEl = document.createElement('label');
        labelEl.textContent = label;
        control.appendChild(labelEl);

        const sliderContainer = document.createElement('div');
        sliderContainer.className = 'slider-container';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'param-slider';
        slider.id = `hf-${paramName}`;
        slider.min = min;
        slider.max = max;
        slider.step = step;
        slider.value = currentValue ?? min;

        const valueDisplay = document.createElement('span');
        valueDisplay.className = 'slider-value';
        valueDisplay.textContent = currentValue ?? min;

        slider.addEventListener('input', () => {
            valueDisplay.textContent = slider.value;
        });

        sliderContainer.appendChild(slider);
        sliderContainer.appendChild(valueDisplay);
        control.appendChild(sliderContainer);

        if (description) {
            const desc = document.createElement('p');
            desc.className = 'param-description';
            desc.textContent = description;
            control.appendChild(desc);
        }

        return control;
    },

    // ========== Qwen3-ASR Pipeline Customization ==========

    async openQwenCustomize(passKey) {
        this.state.currentCustomize = passKey;
        const passState = this.state[passKey];

        try {
            // Get Qwen parameter schema from API
            const result = await pywebview.api.get_qwen_schema();

            if (!result.success) {
                ErrorHandler.show('Error', 'Failed to load Qwen3-ASR parameters: ' + (result.error || 'Unknown error'));
                return;
            }

            // Set modal title
            const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
            const customStatus = passState.customized ? ' [Custom]' : ' [Default]';
            document.getElementById('customizeModalTitle').textContent =
                `${passLabel} Settings (Qwen3-ASR)${customStatus}`;

            // Store schema for reset functionality
            this._qwenSchema = result.schema;

            // Get current values (use stored params or defaults)
            const currentValues = passState.customized && passState.params
                ? { ...passState.params }
                : { ...QwenManager.defaults };

            // Get scene detector from main dropdown
            currentValues.scene = passState.sceneDetector || 'semantic';

            // Generate Qwen-specific tabs
            this.generateQwenTabs(result.schema, currentValues);

            // Show modal
            document.getElementById('customizeModal').classList.add('active');

        } catch (error) {
            ErrorHandler.show('Error', 'Failed to open Qwen3-ASR customize dialog: ' + error);
        }
    },

    generateQwenTabs(schema, currentValues) {
        // Update tab labels for Qwen context
        const tabLabels = {
            'model': 'Model',
            'quality': 'Quality',
            'segmenter': 'Aligner',
            'enhancer': 'Post-Process',
            'scene': 'Scene'
        };

        // Clear all tab panels and update labels
        const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
        tabs.forEach(tab => {
            const panel = document.getElementById(`tab-${tab}`);
            if (panel) panel.innerHTML = '';
            const tabBtn = document.querySelector(`[data-tab="${tab}"]`);
            if (tabBtn) {
                tabBtn.textContent = tabLabels[tab];
                tabBtn.style.display = '';
            }
        });

        // Generate Model tab
        this.generateQwenModelTab('tab-model', schema.model, currentValues);

        // Generate Quality tab (batch_size, max_new_tokens, language)
        this.generateQwenQualityTab('tab-quality', schema.quality, currentValues);

        // Generate Aligner tab (in Segmenter slot - use_aligner, aligner_id)
        this.generateQwenAlignerTab('tab-segmenter', schema.aligner, currentValues);

        // Generate Post-Processing tab (in Enhancer slot)
        this.generateQwenPostProcessTab('tab-enhancer', schema.postprocess, currentValues);

        // Generate Scene tab
        this.generateQwenSceneTab('tab-scene', schema.scene, currentValues);

        // Reset to first tab
        this.switchModalTab('model');
    },

    generateQwenModelTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Model ID dropdown
        const modelDef = schemaSection.model_id;
        container.appendChild(this.createTransformersDropdown(
            'model_id', 'ASR Model',
            modelDef.options,
            currentValues.model_id || modelDef.default,
            '1.7B is more accurate, 0.6B is faster and uses less VRAM'
        ));

        // Device dropdown
        const deviceDef = schemaSection.device;
        container.appendChild(this.createTransformersDropdown(
            'device', 'Device',
            deviceDef.options,
            currentValues.device || deviceDef.default,
            'Compute device (auto will detect GPU availability)'
        ));

        // Data type dropdown
        const dtypeDef = schemaSection.dtype;
        container.appendChild(this.createTransformersDropdown(
            'dtype', 'Data Type',
            dtypeDef.options,
            currentValues.dtype || dtypeDef.default,
            'Model precision (auto selects based on hardware)'
        ));

        // Attention implementation
        const attnDef = schemaSection.attn_implementation;
        container.appendChild(this.createTransformersDropdown(
            'attn_implementation', 'Attention',
            attnDef.options,
            currentValues.attn_implementation || attnDef.default,
            'Attention implementation (sdpa is fastest on most GPUs)'
        ));
    },

    generateQwenQualityTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Batch size slider
        const batchDef = schemaSection.batch_size;
        container.appendChild(this.createTransformersSlider(
            'batch_size', batchDef.label,
            batchDef.min, batchDef.max, batchDef.step,
            currentValues.batch_size ?? batchDef.default,
            'Batch size 1 is recommended for best accuracy'
        ));

        // Max new tokens slider
        const tokensDef = schemaSection.max_new_tokens;
        container.appendChild(this.createTransformersSlider(
            'max_new_tokens', tokensDef.label,
            tokensDef.min, tokensDef.max, tokensDef.step,
            currentValues.max_new_tokens ?? tokensDef.default,
            'Maximum tokens per segment (4096 covers ~5-10 min audio)'
        ));

        // Language dropdown
        const langDef = schemaSection.language;
        container.appendChild(this.createTransformersDropdown(
            'language', 'Language',
            langDef.options,
            currentValues.language || langDef.default,
            'Force language or auto-detect (null = auto)'
        ));
    },

    generateQwenAlignerTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Use aligner toggle
        const alignerToggle = document.createElement('div');
        alignerToggle.className = 'param-control';
        alignerToggle.dataset.param = 'use_aligner';

        const toggleLabel = document.createElement('label');
        toggleLabel.className = 'checkbox-label';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = 'hf-use_aligner';
        checkbox.checked = currentValues.use_aligner !== false;

        const span = document.createElement('span');
        span.textContent = 'Use ForcedAligner for word-level timestamps';

        toggleLabel.appendChild(checkbox);
        toggleLabel.appendChild(span);
        alignerToggle.appendChild(toggleLabel);

        const toggleDesc = document.createElement('p');
        toggleDesc.className = 'param-description';
        toggleDesc.textContent = 'ForcedAligner provides precise word timestamps for better subtitle timing';
        alignerToggle.appendChild(toggleDesc);

        container.appendChild(alignerToggle);

        // Aligner model dropdown
        const alignerDef = schemaSection.aligner_id;
        container.appendChild(this.createTransformersDropdown(
            'aligner_id', 'Aligner Model',
            alignerDef.options,
            currentValues.aligner_id || alignerDef.default,
            'ForcedAligner model for word-level timestamps'
        ));

        // Note about 3-minute limit
        const noteDiv = document.createElement('div');
        noteDiv.className = 'param-note';
        noteDiv.innerHTML = '<strong>Note:</strong> ForcedAligner has a 3-minute segment limit. ' +
            'For longer audio, use Scene Detection to split into shorter segments.';
        container.appendChild(noteDiv);
    },

    generateQwenPostProcessTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Japanese post-processing toggle
        const jpToggle = document.createElement('div');
        jpToggle.className = 'param-control';
        jpToggle.dataset.param = 'japanese_postprocess';

        const toggleLabel = document.createElement('label');
        toggleLabel.className = 'checkbox-label';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = 'hf-japanese_postprocess';
        checkbox.checked = currentValues.japanese_postprocess !== false;

        const span = document.createElement('span');
        span.textContent = 'Enable Japanese post-processing';

        toggleLabel.appendChild(checkbox);
        toggleLabel.appendChild(span);
        jpToggle.appendChild(toggleLabel);

        const toggleDesc = document.createElement('p');
        toggleDesc.className = 'param-description';
        toggleDesc.textContent = 'Apply Japanese-specific regrouping for better subtitle quality';
        jpToggle.appendChild(toggleDesc);

        container.appendChild(jpToggle);

        // Preset dropdown
        const presetDef = schemaSection.postprocess_preset;
        container.appendChild(this.createTransformersDropdown(
            'postprocess_preset', 'Preset',
            presetDef.options,
            currentValues.postprocess_preset || presetDef.default,
            'Regrouping preset optimized for different content types'
        ));
    },

    generateQwenSceneTab(tabId, schemaSection, currentValues) {
        const container = document.getElementById(tabId);

        // Scene detection dropdown (read-only, synced with main dropdown)
        const sceneDef = schemaSection.scene;
        const sceneControl = this.createTransformersDropdown(
            'scene', sceneDef.label,
            sceneDef.options,
            currentValues.scene || sceneDef.default,
            'Scene detection splits audio into segments before processing'
        );

        // Add note that this syncs with main dropdown
        const noteDiv = document.createElement('div');
        noteDiv.className = 'param-note';
        noteDiv.innerHTML = '<strong>Note:</strong> Scene detection is configured in the main Ensemble tab. ' +
            'Qwen3-ASR uses the --qwen-scene parameter internally.';
        sceneControl.appendChild(noteDiv);

        container.appendChild(sceneControl);
    },

    generateModelTab(tabId, currentModel, currentDevice, pipeline) {
        const container = document.getElementById(tabId);

        // Get allowed models for this pipeline
        const allowedModels = this.pipelineModelCompatibility[pipeline] || ['large-v2', 'large-v3'];

        // Model selection
        const modelControl = document.createElement('div');
        modelControl.className = 'param-control';
        modelControl.dataset.param = 'model_name';

        const modelLabel = document.createElement('label');
        modelLabel.textContent = 'Model';
        modelControl.appendChild(modelLabel);

        const modelSelect = document.createElement('select');
        modelSelect.className = 'param-select form-select';
        modelSelect.id = 'model-select';

        const allModelOptions = [
            { value: 'turbo', label: 'Turbo (Fastest)' },
            { value: 'large-v2', label: 'Large-v2 (Balanced)' },
            { value: 'large-v3', label: 'Large-v3 (Latest)' },
            { value: 'kotoba-tech/kotoba-whisper-v2.0-faster', label: 'Kotoba-faster-2.0 (Japanese)' },
            { value: 'RoachLin/kotoba-whisper-v2.2-faster', label: 'Kotoba-faster-2.2 (Japanese)' }
        ];

        // Filter to only allowed models for this pipeline
        const modelOptions = allModelOptions.filter(opt => allowedModels.includes(opt.value));

        // Validate current model - if not allowed, use first allowed
        let selectedModel = currentModel;
        if (!allowedModels.includes(currentModel)) {
            selectedModel = allowedModels[0];
        }

        modelOptions.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.value === selectedModel) option.selected = true;
            modelSelect.appendChild(option);
        });

        modelControl.appendChild(modelSelect);

        const modelDesc = document.createElement('p');
        modelDesc.className = 'param-description';
        if (pipeline === 'fidelity') {
            modelDesc.textContent = 'Whisper model for transcription. Turbo is fastest, Large-v3 is most accurate.';
        } else {
            modelDesc.textContent = 'Whisper model for transcription. Large-v3 is most accurate. (Turbo not supported with faster-whisper)';
        }
        modelControl.appendChild(modelDesc);

        container.appendChild(modelControl);

        // Device selection
        const deviceControl = document.createElement('div');
        deviceControl.className = 'param-control';
        deviceControl.dataset.param = 'device';

        const deviceLabel = document.createElement('label');
        deviceLabel.textContent = 'Device';
        deviceControl.appendChild(deviceLabel);

        const deviceSelect = document.createElement('select');
        deviceSelect.className = 'param-select form-select';
        deviceSelect.id = 'device-select';

        const deviceOptions = [
            { value: 'cuda', label: 'CUDA (GPU)' },
            { value: 'cpu', label: 'CPU' }
        ];

        deviceOptions.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.value === currentDevice) option.selected = true;
            deviceSelect.appendChild(option);
        });

        deviceControl.appendChild(deviceSelect);

        const deviceDesc = document.createElement('p');
        deviceDesc.className = 'param-description';
        deviceDesc.textContent = 'Processing device. CUDA (GPU) is much faster if available.';
        deviceControl.appendChild(deviceDesc);

        container.appendChild(deviceControl);
    },

    switchModalTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.modal-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update tab panels
        document.querySelectorAll('.modal-tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `tab-${tabName}`);
        });
    },

    // AUDIT FIX: Enhanced type inference that checks metadata first
    inferType(value, paramName = null) {
        // Check metadata for explicit type first (most reliable)
        if (paramName && this.parameterMetadata[paramName]) {
            const metadata = this.parameterMetadata[paramName];
            if (metadata.type) {
                return metadata.type === 'boolean' ? 'bool' : metadata.type;
            }
        }

        // Fall back to runtime type inference
        if (typeof value === 'boolean') return 'bool';
        if (typeof value === 'number') {
            return Number.isInteger(value) ? 'int' : 'float';
        }
        if (typeof value === 'string') return 'str';
        if (Array.isArray(value)) return 'array';
        return 'str';
    },

    generateParamControl(param, value, defaultValue) {
        const container = document.createElement('div');
        container.className = 'param-control';
        container.dataset.param = param.name;
        container.dataset.originalType = param.type;  // Store original type for conversion

        const label = document.createElement('label');
        label.textContent = param.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        container.appendChild(label);

        let input;
        const type = param.type;
        const constraints = param.constraints || {};

        // AUDIT FIX: Special handling for array-capable parameters (temperature, suppress_tokens)
        const arrayCapableParams = ['temperature', 'suppress_tokens'];
        if (arrayCapableParams.includes(param.name)) {
            // Create text input with format hint for array values
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'param-text param-array-input';

            // Display value
            const displayValue = value !== undefined ? value : defaultValue;
            if (Array.isArray(displayValue)) {
                input.value = JSON.stringify(displayValue);
            } else if (displayValue !== null && displayValue !== undefined) {
                // Single value - show as-is, will be parsed appropriately
                input.value = String(displayValue);
            } else {
                input.value = '';
            }

            // Set placeholder with format hint
            if (param.name === 'temperature') {
                input.placeholder = 'e.g., [0.0, 0.2, 0.4] or 0.0';
                input.title = 'Temperature for sampling. Use array for fallback temperatures: [0.0, 0.2, 0.4]';
            } else if (param.name === 'suppress_tokens') {
                input.placeholder = 'e.g., [1, 2, 3] or leave empty';
                input.title = 'Token IDs to suppress. Use array format: [1, 2, 3]. Leave empty for default.';
            }

            // Mark as array type for proper parsing
            container.dataset.originalType = 'array';

            container.appendChild(input);

            // Add format hint description
            const hint = document.createElement('p');
            hint.className = 'param-description param-format-hint';
            hint.textContent = param.name === 'temperature'
                ? 'Single value or array of fallback temperatures (e.g., [0.0, 0.2, 0.4])'
                : 'Array of token IDs to suppress (e.g., [1, 2, 3]). Empty for defaults.';
            container.appendChild(hint);

            return container;
        }

        if (type === 'bool' || type === 'boolean') {
            // Checkbox for boolean
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value !== undefined ? value : defaultValue;
            input.className = 'param-checkbox';
        } else if (type === 'float' || type === 'number') {
            // Slider + number input for float
            const wrapper = document.createElement('div');
            wrapper.className = 'param-slider-wrapper';

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.className = 'param-slider';
            slider.min = constraints.ge !== undefined ? constraints.ge : 0;
            slider.max = constraints.le !== undefined ? constraints.le : 1;
            slider.step = constraints.step !== undefined ? constraints.step : (type === 'float' ? 0.01 : 1);
            slider.value = value !== undefined ? value : defaultValue;

            const numberInput = document.createElement('input');
            numberInput.type = 'number';
            numberInput.className = 'param-number';
            numberInput.min = slider.min;
            numberInput.max = slider.max;
            numberInput.step = slider.step;
            numberInput.value = slider.value;

            // Sync slider and number
            slider.addEventListener('input', () => {
                numberInput.value = slider.value;
            });
            numberInput.addEventListener('input', () => {
                slider.value = numberInput.value;
            });

            wrapper.appendChild(slider);
            wrapper.appendChild(numberInput);
            input = wrapper;
            input._getValue = () => parseFloat(numberInput.value);
        } else if (type === 'int' || type === 'integer') {
            // Number input for int
            input = document.createElement('input');
            input.type = 'number';
            input.className = 'param-number';
            input.min = constraints.ge !== undefined ? constraints.ge : 0;
            input.max = constraints.le !== undefined ? constraints.le : 9999;
            input.step = 1;
            input.value = value !== undefined ? value : defaultValue;
        } else if (type === 'str' || type === 'string') {
            // Text input for string
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'param-text';
            input.value = value !== undefined ? value : (defaultValue || '');
        } else if (Array.isArray(param.enum)) {
            // Dropdown for enum
            input = document.createElement('select');
            input.className = 'param-select';
            param.enum.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                if (opt === value || opt === defaultValue) option.selected = true;
                input.appendChild(option);
            });
        } else {
            // Default to text
            input = document.createElement('input');
            input.type = 'text';
            input.className = 'param-text';
            // Use JSON.stringify for arrays to preserve brackets, String for others
            const displayValue = value !== undefined ? value : (defaultValue || '');
            if (Array.isArray(displayValue)) {
                input.value = JSON.stringify(displayValue);
            } else {
                input.value = String(displayValue);
            }
        }

        if (input.tagName) {
            input.dataset.default = defaultValue;
            container.appendChild(input);
        } else {
            // For wrapper elements
            input.dataset = { default: defaultValue };
            container.appendChild(input);
        }

        // Description
        if (param.description) {
            const desc = document.createElement('p');
            desc.className = 'param-description';
            desc.textContent = param.description;
            container.appendChild(desc);
        }

        return container;
    },

    // AUDIT FIX: Validate and clamp parameter values against metadata constraints
    validateAndClampValue(paramName, value, originalType) {
        const metadata = this.parameterMetadata[paramName];
        if (!metadata) return value;  // No metadata, pass through

        // Skip validation for arrays (temperature can be array)
        if (Array.isArray(value)) {
            // Validate each element in array if it's numeric
            if (metadata.ge !== undefined || metadata.le !== undefined) {
                return value.map(v => {
                    if (typeof v === 'number') {
                        let clamped = v;
                        if (metadata.ge !== undefined && v < metadata.ge) clamped = metadata.ge;
                        if (metadata.le !== undefined && v > metadata.le) clamped = metadata.le;
                        return clamped;
                    }
                    return v;
                });
            }
            return value;
        }

        // Validate numeric values
        if (typeof value === 'number' && !isNaN(value)) {
            let clamped = value;
            let wasClamped = false;

            if (metadata.ge !== undefined && value < metadata.ge) {
                clamped = metadata.ge;
                wasClamped = true;
            }
            if (metadata.le !== undefined && value > metadata.le) {
                clamped = metadata.le;
                wasClamped = true;
            }

            if (wasClamped) {
                console.warn(`Parameter '${paramName}': value ${value} clamped to ${clamped} (range: ${metadata.ge}-${metadata.le})`);
            }

            // Ensure integer types stay as integers
            if (originalType === 'int' || originalType === 'integer') {
                return Math.round(clamped);
            }
            return clamped;
        }

        // Convert empty strings to null for optional string params
        if (typeof value === 'string' && value.trim() === '') {
            const optionalStringParams = ['initial_prompt', 'prefix', 'hotwords', 'prompt', 'prepend_punctuations', 'append_punctuations', 'clip_timestamps'];
            if (optionalStringParams.includes(paramName)) {
                return null;
            }
        }

        return value;
    },

    applyCustomization(closeAfter = false) {
        try {
            const passKey = this.state.currentCustomize;
            if (!passKey) {
                ConsoleManager.log('Apply failed: No pass selected for customization', 'error');
                return false;
            }

            const passState = this.state[passKey];
            if (!passState) {
                ConsoleManager.log(`Apply failed: Invalid pass key "${passKey}"`, 'error');
                return false;
            }

            // Handle Transformers separately
            if (passState.isTransformers) {
                return this.applyTransformersCustomization(passKey, closeAfter);
            }

        // Non-Transformers pipeline customization (new tab structure)
        const fullParams = {};
        const validationWarnings = [];

        // Collect ALL values from all tab panels (new structure: model, quality, segmenter, enhancer, scene)
        const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
        tabs.forEach(tab => {
            const panel = document.getElementById(`tab-${tab}`);
            if (!panel) return;

            // Collect standard param-control values
            panel.querySelectorAll('.param-control').forEach(control => {
                const paramName = control.dataset.param;
                const originalType = control.dataset.originalType;
                let value;

                // Find the input element
                const checkbox = control.querySelector('.param-checkbox');
                const slider = control.querySelector('.param-slider');
                const number = control.querySelector('.param-number');
                const text = control.querySelector('.param-text');
                const select = control.querySelector('.param-select');

                if (checkbox) {
                    value = checkbox.checked;
                } else if (slider && number) {
                    // Slider with linked number input
                    value = parseFloat(number.value);
                } else if (slider) {
                    // Slider without number input (fallback)
                    value = parseFloat(slider.value);
                } else if (number) {
                    // Use original type for proper conversion
                    if (originalType === 'int' || originalType === 'integer') {
                        value = parseInt(number.value);
                    } else {
                        value = parseFloat(number.value);
                    }
                } else if (text) {
                    // Convert back to original type
                    const rawValue = text.value;
                    if (originalType === 'array') {
                        // Parse array from string representation
                        try {
                            // Try JSON parse first (handles "[0, 0.2, 0.4]")
                            const parsed = JSON.parse(rawValue);
                            // Ensure result is actually an array
                            if (Array.isArray(parsed)) {
                                value = parsed;
                            } else {
                                // Single value parsed - wrap in array
                                value = [parsed];
                            }
                        } catch {
                            // Fallback: split by comma and convert to numbers
                            if (rawValue.trim()) {
                                value = rawValue.split(',').map(s => {
                                    const trimmed = s.trim();
                                    const num = parseFloat(trimmed);
                                    return isNaN(num) ? trimmed : num;
                                });
                            } else {
                                // Empty string for array - use default or empty array
                                value = [];
                            }
                        }
                    } else if (originalType === 'int' || originalType === 'integer') {
                        value = parseInt(rawValue) || 0;
                    } else if (originalType === 'float' || originalType === 'number') {
                        value = parseFloat(rawValue) || 0;
                    } else {
                        value = rawValue;
                    }
                } else if (select) {
                    // Convert select values based on original type
                    const rawValue = select.value;
                    if (originalType === 'int' || originalType === 'integer') {
                        value = parseInt(rawValue);
                    } else if (originalType === 'float' || originalType === 'number') {
                        value = parseFloat(rawValue);
                    } else {
                        value = rawValue;
                    }
                }

                // AUDIT FIX: Validate and clamp value against metadata constraints
                if (value !== undefined) {
                    const originalValue = value;
                    value = this.validateAndClampValue(paramName, value, originalType);

                    // Track if value was modified
                    if (value !== originalValue && typeof originalValue === 'number') {
                        validationWarnings.push(`${paramName}: ${originalValue} -> ${value}`);
                    }

                    // Store validated value (skip null for optional params that were empty)
                    if (value !== null) {
                        fullParams[paramName] = value;
                    }
                }
            });

            // Collect backend selector values (used in segmenter, enhancer, scene tabs)
            panel.querySelectorAll('.backend-selector').forEach(select => {
                const paramName = select.dataset.param;
                if (paramName && select.value) {
                    fullParams[paramName] = select.value;
                }
            });

            // Collect checkbox-list values (used for FFmpeg DSP effects)
            panel.querySelectorAll('.checkbox-list').forEach(list => {
                const paramName = list.dataset.param;
                if (paramName) {
                    const selectedValues = [];
                    list.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
                        selectedValues.push(cb.value);
                    });
                    fullParams[paramName] = selectedValues;
                }
            });
        });

        // Log validation warnings if any
        if (validationWarnings.length > 0) {
            ConsoleManager.log(`Clamped ${validationWarnings.length} out-of-range values`, 'warn');
            console.log('Validation adjustments:', validationWarnings);
        }

        // Store full params in pass state and mark as customized
        this.state[passKey].params = fullParams;
        this.state[passKey].customized = true;

        // Sync scene_detection_method to state and dropdown (bidirectional sync)
        if (fullParams.scene_detection_method) {
            this.state[passKey].sceneDetector = fullParams.scene_detection_method;
            // Also update the Ensemble tab dropdown UI
            const dropdownId = passKey === 'pass1' ? 'pass1-scene' : 'pass2-scene';
            const dropdown = document.getElementById(dropdownId);
            if (dropdown) dropdown.value = fullParams.scene_detection_method;
        }

        const paramCount = Object.keys(fullParams).length;
        const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
        ConsoleManager.log(`Saved ${paramCount} parameters for ${passLabel} (Custom)`, 'info');

        this.updateBadges();
        this.showApplyFeedback();

        if (closeAfter) {
            this.closeModal();
        }
        return true;

        } catch (error) {
            ConsoleManager.log(`Apply failed: ${error.message}`, 'error');
            console.error('applyCustomization error:', error);
            return false;
        }
    },

    applyTransformersCustomization(passKey, closeAfter = false) {
        try {
            // Collect Transformers-specific parameters from the new tab structure
            const hfParams = {};
            const passState = this.state[passKey];

        // Collect ALL values from tab panels using the same approach as non-Transformers
        const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
        tabs.forEach(tab => {
            const panel = document.getElementById(`tab-${tab}`);
            if (!panel) return;

            // Collect standard param-control values
            panel.querySelectorAll('.param-control').forEach(control => {
                const paramName = control.dataset.param;
                const originalType = control.dataset.originalType;
                let value;

                const checkbox = control.querySelector('.param-checkbox');
                const slider = control.querySelector('.param-slider');
                const number = control.querySelector('.param-number');
                const text = control.querySelector('.param-text');
                const select = control.querySelector('.param-select');

                if (checkbox) {
                    value = checkbox.checked;
                } else if (slider && number) {
                    // Slider with linked number input
                    value = parseFloat(number.value);
                } else if (slider) {
                    // Slider without number input (fallback)
                    value = parseFloat(slider.value);
                } else if (number) {
                    if (originalType === 'int' || originalType === 'integer') {
                        value = parseInt(number.value);
                    } else {
                        value = parseFloat(number.value);
                    }
                } else if (text) {
                    value = text.value;
                } else if (select) {
                    const rawValue = select.value;
                    if (originalType === 'int' || originalType === 'integer') {
                        value = parseInt(rawValue);
                    } else if (originalType === 'float' || originalType === 'number') {
                        value = parseFloat(rawValue);
                    } else {
                        value = rawValue;
                    }
                }

                if (value !== undefined && value !== null) {
                    hfParams[paramName] = value;
                }
            });

            // Collect backend selector values
            panel.querySelectorAll('.backend-selector').forEach(select => {
                const paramName = select.dataset.param;
                if (paramName && select.value) {
                    hfParams[paramName] = select.value;
                }
            });

            // Collect checkbox-list values (for FFmpeg DSP effects)
            panel.querySelectorAll('.checkbox-list').forEach(list => {
                const paramName = list.dataset.param;
                if (paramName) {
                    const selectedValues = [];
                    list.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
                        selectedValues.push(cb.value);
                    });
                    hfParams[paramName] = selectedValues;
                }
            });
        });

        // Store params
        this.state[passKey].params = hfParams;
        this.state[passKey].customized = true;

        const paramCount = Object.keys(hfParams).length;
        const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
        ConsoleManager.log(`Saved ${paramCount} Transformers parameters for ${passLabel} (Custom)`, 'info');

        this.updateBadges();
        this.showApplyFeedback();

        if (closeAfter) {
            this.closeModal();
        }
        return true;

        } catch (error) {
            ConsoleManager.log(`Apply Transformers failed: ${error.message}`, 'error');
            console.error('applyTransformersCustomization error:', error);
            return false;
        }
    },

    showApplyFeedback() {
        // Show visual feedback on Apply button to indicate success
        const applyBtn = document.getElementById('customizeModalApply');
        if (!applyBtn) return;

        // Store original state
        const originalText = applyBtn.textContent;
        const originalClass = applyBtn.className;

        // Show success state
        applyBtn.textContent = 'Applied';
        applyBtn.className = 'btn btn-success';

        // Revert after 1.5 seconds
        setTimeout(() => {
            applyBtn.textContent = originalText;
            applyBtn.className = originalClass;
        }, 1500);
    },

    async resetToDefaults() {
        if (!this.state.currentCustomize) return;
        const passKey = this.state.currentCustomize;
        const passState = this.state[passKey];

        // Handle Transformers separately
        if (passState.isTransformers) {
            this.resetTransformersToDefaults(passKey);
            return;
        }

        // Use stored defaults from openCustomize
        if (!this._currentDefaults) {
            ErrorHandler.show('Error', 'No defaults available');
            return;
        }

        // Flatten all defaults including model
        const defaults = {
            model_name: this._currentDefaults.model || 'large-v2',
            device: 'cuda',
            ...this._currentDefaults.transcriber,
            ...this._currentDefaults.decoder,
            ...this._currentDefaults.engine,
            ...this._currentDefaults.vad,
            ...this._currentDefaults.scene
        };

        // Reset all controls in all tabs (new structure: model, quality, segmenter, enhancer, scene)
        const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
        tabs.forEach(tab => {
            const panel = document.getElementById(`tab-${tab}`);
            if (!panel) return;

            // Reset standard param-control values
            panel.querySelectorAll('.param-control').forEach(control => {
                const paramName = control.dataset.param;
                const defaultValue = defaults[paramName];

                const checkbox = control.querySelector('.param-checkbox');
                const slider = control.querySelector('.param-slider');
                const number = control.querySelector('.param-number');
                const text = control.querySelector('.param-text');
                const select = control.querySelector('.param-select');

                if (checkbox) {
                    checkbox.checked = defaultValue;
                } else if (slider) {
                    slider.value = defaultValue;
                    control.querySelector('.param-number').value = defaultValue;
                } else if (number) {
                    number.value = defaultValue;
                } else if (text) {
                    text.value = defaultValue || '';
                } else if (select) {
                    select.value = defaultValue;
                }
            });

            // Reset backend selectors to default values
            panel.querySelectorAll('.backend-selector').forEach(select => {
                const paramName = select.dataset.param;
                if (paramName && defaults[paramName]) {
                    select.value = defaults[paramName];
                }
            });

            // Reset checkbox lists (e.g., DSP effects) - we'll use the passState default
            panel.querySelectorAll('.checkbox-list').forEach(list => {
                const paramName = list.dataset.param;
                const defaultEffects = paramName === 'dspEffects' ? ['loudnorm'] : [];
                list.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                    cb.checked = defaultEffects.includes(cb.value);
                });
            });
        });

        // Clear customized state - will use defaults
        passState.params = null;
        passState.customized = false;

        const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
        ConsoleManager.log(`Reset ${passLabel} to defaults`, 'info');

        this.updateBadges();
    },

    resetTransformersToDefaults(passKey) {
        // Reset Transformers controls using the new tab structure
        const defaults = TransformersManager.defaults || {};

        // Reset all controls in all tabs (new structure)
        const tabs = ['model', 'quality', 'segmenter', 'enhancer', 'scene'];
        tabs.forEach(tab => {
            const panel = document.getElementById(`tab-${tab}`);
            if (!panel) return;

            // Reset standard param-control values
            panel.querySelectorAll('.param-control').forEach(control => {
                const paramName = control.dataset.param;
                const defaultValue = defaults[paramName];

                const checkbox = control.querySelector('.param-checkbox');
                const slider = control.querySelector('.param-slider');
                const number = control.querySelector('.param-number');
                const text = control.querySelector('.param-text');
                const select = control.querySelector('.param-select');

                if (checkbox && defaultValue !== undefined) {
                    checkbox.checked = defaultValue;
                } else if (slider && defaultValue !== undefined) {
                    slider.value = defaultValue;
                    const numInput = control.querySelector('.param-number');
                    if (numInput) numInput.value = defaultValue;
                    const valueDisplay = slider.parentElement?.querySelector('.slider-value');
                    if (valueDisplay) valueDisplay.textContent = defaultValue;
                } else if (number && defaultValue !== undefined) {
                    number.value = defaultValue;
                } else if (text) {
                    text.value = defaultValue || '';
                } else if (select && defaultValue !== undefined) {
                    select.value = defaultValue;
                }
            });

            // Reset backend selectors
            panel.querySelectorAll('.backend-selector').forEach(select => {
                const paramName = select.dataset.param;
                if (paramName && defaults[paramName]) {
                    select.value = defaults[paramName];
                }
            });

            // Reset checkbox lists
            panel.querySelectorAll('.checkbox-list').forEach(list => {
                const paramName = list.dataset.param;
                const defaultEffects = paramName === 'dspEffects' ? ['loudnorm'] : [];
                list.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                    cb.checked = defaultEffects.includes(cb.value);
                });
            });
        });

        // Clear customized state
        this.state[passKey].params = null;
        this.state[passKey].customized = false;

        const passLabel = passKey === 'pass1' ? 'Pass 1' : 'Pass 2';
        ConsoleManager.log(`Reset ${passLabel} Transformers parameters to defaults`, 'info');

        this.updateBadges();
    },

    closeModal() {
        try {
            const modal = document.getElementById('customizeModal');
            if (modal) {
                modal.classList.remove('active');
            }
            this.state.currentCustomize = null;
        } catch (error) {
            ConsoleManager.log(`Close modal failed: ${error.message}`, 'error');
            console.error('closeModal error:', error);
        }
    },

    okAndClose() {
        // OK button: Apply changes and close modal
        const success = this.applyCustomization(true);  // true = close after apply
        // If apply failed, still try to close modal to avoid trapping user
        if (!success) {
            this.closeModal();
        }
    },

    collectConfig() {
        // Build two-pass ensemble configuration for API
        // Uses Full Configuration Snapshot approach:
        // - If customized: send full params (backend uses as-is)
        // - If not customized: send pipeline+sensitivity (backend resolves defaults)

        // Helpers for pipeline-specific null handling:
        // - Sensitivity: null for both Transformers AND Qwen (neither uses sensitivity presets)
        // - Segmenter: null for Transformers only (Qwen uses segmenter as post-ASR VAD filter)
        const disableSensitivity = (passState) => passState.isTransformers || passState.isQwen;
        const disableSegmenter = (passState) => passState.isTransformers;  // NOT Qwen!

        const config = {
            inputs: AppState.selectedFiles,
            output_dir: document.getElementById('outputDir').value,
            pass1: {
                pipeline: this.state.pass1.pipeline,
                sensitivity: disableSensitivity(this.state.pass1) ? null : this.state.pass1.sensitivity,
                sceneDetector: this.state.pass1.sceneDetector,
                speechEnhancer: this.state.pass1.speechEnhancer,
                dspEffects: this.state.pass1.speechEnhancer === 'ffmpeg-dsp' ? this.state.pass1.dspEffects : null,
                speechSegmenter: disableSegmenter(this.state.pass1) ? null : this.state.pass1.speechSegmenter,
                model: this.state.pass1.model,
                customized: this.state.pass1.customized,
                params: this.state.pass1.customized ? this.state.pass1.params : null,
                isTransformers: this.state.pass1.isTransformers,
                isQwen: this.state.pass1.isQwen
            },
            pass2: {
                enabled: this.state.pass2.enabled,
                pipeline: this.state.pass2.pipeline,
                sensitivity: disableSensitivity(this.state.pass2) ? null : this.state.pass2.sensitivity,
                sceneDetector: this.state.pass2.sceneDetector,
                speechEnhancer: this.state.pass2.speechEnhancer,
                dspEffects: this.state.pass2.speechEnhancer === 'ffmpeg-dsp' ? this.state.pass2.dspEffects : null,
                speechSegmenter: disableSegmenter(this.state.pass2) ? null : this.state.pass2.speechSegmenter,
                model: this.state.pass2.model,
                customized: this.state.pass2.customized,
                params: this.state.pass2.customized ? this.state.pass2.params : null,
                isTransformers: this.state.pass2.isTransformers,
                isQwen: this.state.pass2.isQwen
            },
            merge_strategy: this.state.mergeStrategy,
            source_language: document.getElementById('source-language').value,
            subs_language: document.getElementById('language').value,
            debug: document.getElementById('debugLogging').checked,
            keep_temp: document.getElementById('keepTemp').checked,
            temp_dir: document.getElementById('tempDir').value.trim()
        };

        // Add translation settings if enabled (single CLI command approach)
        if (TranslateIntegrationManager.isEnabled()) {
            const translateSettings = TranslateIntegrationManager.getSettings();
            config.translate = true;
            config.translate_provider = translateSettings.provider || 'local';
            config.translate_target = translateSettings.target || 'english';
            config.translate_tone = translateSettings.tone || 'standard';
            // Use model override if provided, otherwise use selected model
            config.translate_model = translateSettings.modelOverride || translateSettings.model || null;
            // Local provider doesn't need API key
            config.translate_api_key = (translateSettings.provider === 'local') ? null : (translateSettings.apiKey || null);
            config.translate_title = translateSettings.movieTitle || null;
            config.translate_actress = translateSettings.actress || null;
            config.translate_plot = translateSettings.plot || null;
        }

        return config;
    },

    async startProcessing() {
        // Validate
        if (AppState.selectedFiles.length === 0) {
            ErrorHandler.show('No Files Selected', 'Please add at least one file or folder before starting.');
            return;
        }

        try {
            const config = this.collectConfig();

            // Capture translation settings before starting (in case user switches tabs)
            TranslateIntegrationManager.captureSettingsOnStart();

            const result = await pywebview.api.start_ensemble_twopass(config);

            if (result.success) {
                // Update UI state
                AppState.isRunning = true;
                ProcessManager.updateButtonStates();

                // Start progress indication
                ProgressManager.setIndeterminate(true);
                ProgressManager.setStatus('Running...');

                // Log command
                if (result.command) {
                    ConsoleManager.appendRaw(`\n> ${result.command}\n`);
                }

                // Start polling
                ProcessManager.startLogPolling();
                ProcessManager.startStatusMonitoring();

                ConsoleManager.log('Two-pass ensemble started', 'info');
            } else {
                ErrorHandler.show('Start Failed', result.message);
            }
        } catch (error) {
            ErrorHandler.show('Start Error', error.toString());
        }
    },

    /**
     * Update segmenter dropdown options based on backend availability.
     * Called on pywebviewready when API becomes available.
     */
    async updateSegmenterAvailability() {
        if (!window.pywebview || !pywebview.api) {
            console.warn('PyWebView API not available for segmenter check');
            return;
        }

        try {
            const result = await pywebview.api.get_speech_segmenter_backends();
            if (!result.success) {
                console.error('Failed to get segmenter backends:', result.error);
                return;
            }

            const backends = result.backends;
            const availabilityMap = {};
            backends.forEach(b => {
                availabilityMap[b.name] = {
                    available: b.available,
                    hint: b.install_hint || ''
                };
            });

            // Update both pass1 and pass2 segmenter dropdowns
            ['pass1-segmenter', 'pass2-segmenter'].forEach(selectId => {
                const select = document.getElementById(selectId);
                if (!select) return;

                Array.from(select.options).forEach(option => {
                    const backend = option.value;
                    // Skip 'none' - always available
                    if (backend === 'none' || backend === '') return;

                    // Normalize backend name for lookup (silero-v4.0 -> silero)
                    const lookupName = backend === 'silero-v4.0' ? 'silero' : backend;
                    const info = availabilityMap[lookupName];

                    if (info && !info.available) {
                        option.disabled = true;
                        option.title = info.hint || 'Not available';
                        option.textContent = option.textContent.replace(/ \(N\/A\)$/, '') + ' (N/A)';
                    } else {
                        option.disabled = false;
                        option.title = '';
                        option.textContent = option.textContent.replace(/ \(N\/A\)$/, '');
                    }
                });
            });

            console.log('Segmenter availability updated');
        } catch (error) {
            console.error('Error updating segmenter availability:', error);
        }
    },

    /**
     * Update enhancer dropdown options based on backend availability.
     * Called on pywebviewready when API becomes available.
     * Handles both legacy format ("clearvoice") and new format ("clearvoice:MossFormer2_SE_48K").
     */
    async updateEnhancerAvailability() {
        if (!window.pywebview || !pywebview.api) {
            console.warn('PyWebView API not available for enhancer check');
            return;
        }

        try {
            const result = await pywebview.api.get_speech_enhancer_backends();
            if (!result.success) {
                console.error('Failed to get enhancer backends:', result.error);
                return;
            }

            const backends = result.backends;
            const availabilityMap = {};
            backends.forEach(b => {
                availabilityMap[b.name] = {
                    available: b.available,
                    hint: b.install_hint || ''
                };
            });

            // Update both pass1 and pass2 enhancer dropdowns
            ['pass1-enhancer', 'pass2-enhancer'].forEach(selectId => {
                const select = document.getElementById(selectId);
                if (!select) return;

                Array.from(select.options).forEach(option => {
                    const value = option.value;
                    // Skip 'none' - always available
                    if (value === 'none' || value === '') return;

                    // Extract backend name from "backend:model" format
                    const backend = value.includes(':') ? value.split(':')[0] : value;
                    const info = availabilityMap[backend];

                    if (info && !info.available) {
                        option.disabled = true;
                        option.title = info.hint || 'Not available';
                        option.textContent = option.textContent.replace(/ \(N\/A\)$/, '') + ' (N/A)';
                    } else {
                        option.disabled = false;
                        option.title = '';
                        option.textContent = option.textContent.replace(/ \(N\/A\)$/, '');
                    }
                });
            });

            console.log('Enhancer availability updated');
        } catch (error) {
            console.error('Error updating enhancer availability:', error);
        }
    }
};

// ============================================================
// Run Controls Integration
// ============================================================
const RunControls = {
    init() {
        document.getElementById('startBtn').addEventListener('click', () => {
            // Check if in Ensemble mode
            if (AppState.activeTab === 'tab3') {
                EnsembleManager.startProcessing();
            } else {
                ProcessManager.start();
            }
        });
        document.getElementById('cancelBtn').addEventListener('click', () => ProcessManager.cancel());
    }
};

// ============================================================
// About Dialog
// ============================================================
async function showAbout() {
    const modal = document.getElementById('aboutModal');
    const versionEl = document.getElementById('aboutVersion');

    // Fetch version from API and update display
    try {
        const result = await pywebview.api.get_version();
        if (result.success) {
            versionEl.textContent = `Version ${result.version}`;
        }
    } catch (e) {
        // Keep existing text on error (fallback to HTML default)
        console.warn('Could not fetch version:', e);
    }

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
// Changelog Manager (release notes display)
// ============================================================
const ChangelogManager = {
    releaseUrl: null,
    cachedNotes: null,

    async show(version = null) {
        const modal = document.getElementById('changelogModal');
        const loading = document.getElementById('changelogLoading');
        const content = document.getElementById('changelogContent');
        const error = document.getElementById('changelogError');

        // Show modal with loading state
        modal.classList.add('active');
        loading.style.display = 'block';
        content.style.display = 'none';
        error.style.display = 'none';

        try {
            // Get release notes from update check result or fetch fresh
            let notes = null;
            let releaseVersion = version;
            let releaseDate = '';
            let releaseUrl = 'https://github.com/meizhong986/WhisperJAV/releases';

            if (UpdateManager.updateInfo && UpdateManager.updateInfo.version_info) {
                notes = UpdateManager.updateInfo.release_notes;
                releaseVersion = UpdateManager.updateInfo.latest_version;
                releaseUrl = UpdateManager.updateInfo.release_url || releaseUrl;
            } else if (window.pywebview && pywebview.api) {
                // Fetch fresh
                const result = await pywebview.api.check_for_updates(false);
                if (result.success && result.release_notes) {
                    notes = result.release_notes;
                    releaseVersion = result.latest_version;
                    releaseUrl = result.release_url || releaseUrl;
                }
            }

            this.releaseUrl = releaseUrl;

            if (notes) {
                // Parse and display notes
                document.getElementById('changelogVersion').textContent = `v${releaseVersion}`;
                document.getElementById('changelogDate').textContent = releaseDate;
                document.getElementById('changelogNotes').innerHTML = this.parseMarkdown(notes);

                loading.style.display = 'none';
                content.style.display = 'block';
            } else {
                // Show error state
                loading.style.display = 'none';
                error.style.display = 'block';
            }
        } catch (err) {
            console.error('Failed to load changelog:', err);
            loading.style.display = 'none';
            error.style.display = 'block';
        }
    },

    close() {
        const modal = document.getElementById('changelogModal');
        modal.classList.remove('active');
    },

    openGitHub() {
        const url = this.releaseUrl || 'https://github.com/meizhong986/WhisperJAV/releases';
        window.open(url, '_blank');
    },

    // Simple markdown to HTML converter for release notes
    parseMarkdown(text) {
        if (!text) return '';

        // Escape HTML
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Bold and italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Lists
        html = html.replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Paragraphs (lines not already wrapped)
        html = html.replace(/^(?!<[hulo]|<li|<pre)(.+)$/gm, '<p>$1</p>');

        // Clean up empty paragraphs
        html = html.replace(/<p><\/p>/g, '');

        return html;
    }
};

// ============================================================
// Update Manager (version checking and update notifications)
// ============================================================
const UpdateManager = {
    updateInfo: null,
    releaseUrl: null,

    async init() {
        // Bind event handlers
        document.getElementById('updateNowBtn')?.addEventListener('click', () => this.startUpdate());
        document.getElementById('updateDismissBtn')?.addEventListener('click', () => this.dismiss());
        document.getElementById('updateReleaseNotesBtn')?.addEventListener('click', () => this.openReleaseNotes());

        // Check for updates after a short delay (don't block startup)
        setTimeout(() => this.checkForUpdates(), 3000);
    },

    async checkForUpdates(force = false) {
        if (!window.pywebview || !pywebview.api) {
            console.log('PyWebView API not ready for update check');
            return;
        }

        try {
            const result = await pywebview.api.check_for_updates(force);

            if (!result.success) {
                console.warn('Update check failed:', result.error);
                return;
            }

            this.updateInfo = result;

            if (result.update_available) {
                // Check if user dismissed this version
                const dismissed = await this.getDismissedVersion();
                if (dismissed && dismissed.version === result.latest_version) {
                    // Check notification level - always show critical
                    if (result.notification_level !== 'critical') {
                        console.log(`Update ${result.latest_version} was dismissed`);
                        return;
                    }
                }

                this.showBanner(result);
            }
        } catch (error) {
            console.error('Update check error:', error);
        }
    },

    async getDismissedVersion() {
        try {
            const result = await pywebview.api.get_dismissed_update();
            return result.success ? result.dismissed : null;
        } catch (error) {
            return null;
        }
    },

    showBanner(info) {
        const banner = document.getElementById('updateBanner');
        const title = document.getElementById('updateTitle');
        const version = document.getElementById('updateVersion');

        if (!banner || !title || !version) return;

        // Set content based on notification level
        const level = info.notification_level || 'minor';
        let titleText = 'Update Available';

        if (level === 'critical') {
            titleText = 'Critical Update Available';
        } else if (level === 'major') {
            titleText = 'Major Update Available';
        } else if (level === 'minor') {
            titleText = 'New Version Available';
        } else if (level === 'patch') {
            titleText = 'Patch Available';
        }

        title.textContent = titleText;
        version.textContent = `v${info.latest_version} is now available (you have v${info.current_version})`;

        // Store release URL
        this.releaseUrl = info.release_url;

        // Set banner style based on level
        banner.className = 'update-banner visible ' + level;

        console.log(`Showing update banner: ${info.current_version} -> ${info.latest_version} (${level})`);
    },

    hideBanner() {
        const banner = document.getElementById('updateBanner');
        if (banner) {
            banner.classList.remove('visible');
        }
    },

    async dismiss() {
        if (!this.updateInfo) return;

        // Record dismissal
        try {
            await pywebview.api.dismiss_update_notification(this.updateInfo.latest_version);
        } catch (error) {
            console.error('Failed to record dismissal:', error);
        }

        this.hideBanner();
        ConsoleManager.log('Update notification dismissed', 'info');
    },

    openReleaseNotes() {
        // Show the changelog modal instead of opening browser
        ChangelogManager.show();
    },

    async startUpdate() {
        // Confirm with user
        if (!confirm('WhisperJAV will close and update. The application will restart automatically after the update completes.\n\nContinue?')) {
            return;
        }

        // Show progress overlay
        const overlay = document.getElementById('updateProgressOverlay');
        const stepEl = document.getElementById('updateProgressStep');

        if (overlay) {
            overlay.classList.add('visible');
        }

        if (stepEl) {
            stepEl.textContent = 'Starting update process...';
        }

        try {
            // Hide the banner
            this.hideBanner();

            // Start update via API
            const result = await pywebview.api.start_update(false);

            if (result.success) {
                if (stepEl) {
                    stepEl.textContent = 'Closing application for update...';
                }

                ConsoleManager.log('Update started, GUI will close...', 'info');

                // Give user a moment to see the message, then close
                setTimeout(() => {
                    // Close the window - the update wrapper will handle the rest
                    if (window.pywebview && window.pywebview.api) {
                        // Use window close if available
                        window.close();
                    }
                }, 1500);
            } else {
                // Update failed to start
                if (overlay) {
                    overlay.classList.remove('visible');
                }
                ErrorHandler.show('Update Failed', result.error || 'Failed to start update process');
            }
        } catch (error) {
            // Hide overlay on error
            if (overlay) {
                overlay.classList.remove('visible');
            }
            ErrorHandler.show('Update Error', error.toString());
        }
    }
};

// ============================================================
// Update Check Manager (manual update check from menu)
// ============================================================
const UpdateCheckManager = {
    modal: null,

    init() {
        this.modal = document.getElementById('updateCheckModal');
        if (!this.modal) {
            console.warn('UpdateCheckManager: modal not found');
            return;
        }

        // Menu item click
        const checkBtn = document.getElementById('checkUpdatesBtn');
        if (checkBtn) {
            checkBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                document.getElementById('themeMenu')?.classList.remove('active');
                this.show();
            });
        }

        // Modal close handlers
        document.getElementById('updateCheckModalClose')?.addEventListener('click', () => this.close());
        document.getElementById('updateCheckLater')?.addEventListener('click', () => this.close());
        document.getElementById('updateCheckNow')?.addEventListener('click', () => this.startUpdate());
        document.getElementById('updateCheckDownload')?.addEventListener('click', () => this.openDownloadPage());

        // Close on overlay click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) this.close();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal.classList.contains('active')) {
                this.close();
            }
        });
    },

    async show() {
        // Reset and show loading state
        this.showLoading();
        this.modal.classList.add('active');

        try {
            // Force fresh check (bypass cache)
            const result = await pywebview.api.check_for_updates(true);
            this.showResult(result);
        } catch (err) {
            console.error('Update check error:', err);
            this.showError(err);
        }
    },

    // Store the last result for update actions
    lastResult: null,

    showLoading() {
        document.getElementById('updateCheckLoading').style.display = 'block';
        document.getElementById('updateCheckResult').style.display = 'none';
        document.getElementById('updateCheckError').style.display = 'none';
    },

    showResult(result) {
        this.lastResult = result;
        document.getElementById('updateCheckLoading').style.display = 'none';
        document.getElementById('updateCheckResult').style.display = 'block';

        // Set current version info
        document.getElementById('updateCurrentVersion').textContent = 'v' + result.current_version;

        // Show commit hash if available
        const commitEl = document.getElementById('updateCurrentCommit');
        if (commitEl && result.current_commit) {
            commitEl.textContent = '(' + result.current_commit.substring(0, 7) + ')';
            commitEl.style.display = 'inline';
        } else if (commitEl) {
            commitEl.style.display = 'none';
        }

        // === Stable Release Track ===
        this.updateStableTrack(result);

        // === Development Track ===
        this.updateDevTrack(result);
    },

    updateStableTrack(result) {
        const stableUpToDate = document.getElementById('stableUpToDate');
        const stableUpdateAvailable = document.getElementById('stableUpdateAvailable');
        const stableReleaseNotes = document.getElementById('stableReleaseNotes');
        const stableActions = document.getElementById('stableActions');
        const stableLatestVersion = document.getElementById('stableLatestVersion');
        const stableNewVersion = document.getElementById('stableNewVersion');
        const stableBadge = document.getElementById('stableUpdateBadge');

        if (result.update_available && result.latest_version) {
            // Stable update available
            stableUpToDate.style.display = 'none';
            stableUpdateAvailable.style.display = 'flex';
            stableNewVersion.textContent = result.latest_version;

            // Set badge color based on notification level
            const level = result.notification_level || 'patch';
            stableBadge.textContent = level.toUpperCase();
            stableBadge.className = 'update-badge ' + level;

            // Show release notes if available
            if (result.release_notes) {
                const notesContent = document.getElementById('stableReleaseNotesContent');
                notesContent.innerHTML = this.parseMarkdown(result.release_notes);
                stableReleaseNotes.style.display = 'block';
            } else {
                stableReleaseNotes.style.display = 'none';
            }

            // Show action button (Download for major, Update for others)
            if (level === 'major') {
                stableActions.innerHTML = '<button id="updateToStable" class="btn btn-primary btn-sm">Download from GitHub</button>';
                document.getElementById('updateToStable').addEventListener('click', () => this.openDownloadPage());
            } else {
                stableActions.innerHTML = '<button id="updateToStable" class="btn btn-primary btn-sm">Update to Stable</button>';
                document.getElementById('updateToStable').addEventListener('click', () => this.startUpdate('stable'));
            }
            stableActions.style.display = 'block';
        } else {
            // Up to date with stable
            stableUpToDate.style.display = 'flex';
            stableLatestVersion.textContent = result.current_version;
            stableUpdateAvailable.style.display = 'none';
            stableReleaseNotes.style.display = 'none';
            stableActions.style.display = 'none';
        }
    },

    updateDevTrack(result) {
        const devUpToDate = document.getElementById('devUpToDate');
        const devUpdateAvailable = document.getElementById('devUpdateAvailable');
        const devCommitsList = document.getElementById('devCommitsList');
        const devActions = document.getElementById('devActions');
        const devCommitsAhead = document.getElementById('devCommitsAhead');
        const devRecentCommits = document.getElementById('devRecentCommits');

        if (result.dev_update_available && result.dev_commits_ahead > 0) {
            // Dev updates available
            devUpToDate.style.display = 'none';
            devUpdateAvailable.style.display = 'flex';
            devCommitsAhead.textContent = result.dev_commits_ahead;

            // Show recent commits list
            if (result.dev_recent_commits && result.dev_recent_commits.length > 0) {
                devRecentCommits.innerHTML = result.dev_recent_commits
                    .slice(0, 5) // Show max 5 recent commits
                    .map(commit => {
                        const shortSha = commit.short_sha || commit.sha.substring(0, 7);
                        const message = this.escapeHtml(commit.message.split('\n')[0]); // First line only
                        return `<li><code>${shortSha}</code> ${message}</li>`;
                    })
                    .join('');
                devCommitsList.style.display = 'block';
            } else {
                devCommitsList.style.display = 'none';
            }

            // Show action button
            devActions.innerHTML = '<button id="updateToDev" class="btn btn-secondary btn-sm">Update to Latest Dev</button>';
            document.getElementById('updateToDev').addEventListener('click', () => this.startUpdate('dev'));
            devActions.style.display = 'block';
        } else {
            // Up to date with dev
            devUpToDate.style.display = 'flex';
            devUpdateAvailable.style.display = 'none';
            devCommitsList.style.display = 'none';
            devActions.style.display = 'none';
        }
    },

    showError(error) {
        document.getElementById('updateCheckLoading').style.display = 'none';
        document.getElementById('updateCheckResult').style.display = 'none';
        document.getElementById('updateCheckError').style.display = 'block';
    },

    async startUpdate(track = 'stable') {
        // Close modal
        this.close();

        // Use existing UpdateManager to show progress and start update
        // Both tracks use the same update mechanism (pip install from GitHub)
        // The difference is just which version was chosen
        console.log('Starting update from track:', track);
        UpdateManager.startUpdate();
    },

    async openDownloadPage() {
        // Open GitHub releases in browser
        try {
            await pywebview.api.open_url('https://github.com/meizhong986/WhisperJAV/releases');
        } catch (err) {
            // Fallback: open in new window
            window.open('https://github.com/meizhong986/WhisperJAV/releases', '_blank');
        }
        this.close();
    },

    close() {
        this.modal.classList.remove('active');
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    parseMarkdown(text) {
        // Simple markdown to HTML conversion
        if (!text) return '';
        return text
            .replace(/^### (.+)$/gm, '<h4>$1</h4>')
            .replace(/^## (.+)$/gm, '<h3>$1</h3>')
            .replace(/^# (.+)$/gm, '<h2>$1</h2>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/^\* (.+)$/gm, '<li>$1</li>')
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/\n\n/g, '<br><br>');
    }
};

// ============================================================
// Translator Manager (Tab 4 - Standalone Translation)
// ============================================================
const TranslatorManager = {
    // State
    state: {
        files: [],           // {path, name, status: pending|translating|completed|error}
        isRunning: false,
        progress: 0,
        currentFile: null
    },

    // Provider model options
    providerModels: {
        local: ['gemma-9b', 'llama-8b', 'llama-3b', 'auto'],
        deepseek: ['deepseek-chat', 'deepseek-coder'],
        gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
        claude: ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
        gpt: ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
        openrouter: ['deepseek/deepseek-chat', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o'],
        glm: ['glm-4-flash', 'glm-4', 'glm-4-plus'],
        groq: ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768']
    },

    init() {
        // Bind file action buttons
        document.getElementById('translatorAddFiles')?.addEventListener('click', () => this.addFiles());
        document.getElementById('translatorRemoveSelected')?.addEventListener('click', () => this.removeSelected());
        document.getElementById('translatorClearFiles')?.addEventListener('click', () => this.clearFiles());

        // Bind provider change
        document.getElementById('translatorProvider')?.addEventListener('change', (e) => {
            this.updateModelOptions(e.target.value);
            this.updateApiKeyStatus(e.target.value);
        });

        // Bind test connection
        document.getElementById('translatorTestConnection')?.addEventListener('click', () => this.testConnection());

        // Bind control buttons
        document.getElementById('translatorStartBtn')?.addEventListener('click', () => this.startTranslation());
        document.getElementById('translatorCancelBtn')?.addEventListener('click', () => this.cancelTranslation());

        // Initialize model options for default provider (Local LLM)
        this.updateModelOptions('local');
        this.updateApiKeyStatus('local');

        console.log('TranslatorManager initialized');
    },

    /**
     * Open file dialog to add SRT files
     */
    async addFiles() {
        try {
            const result = await pywebview.api.select_srt_files();
            if (result.success && result.files && result.files.length > 0) {
                // Add files that aren't already in the list
                result.files.forEach(filePath => {
                    if (!this.state.files.some(f => f.path === filePath)) {
                        const name = filePath.split(/[\\/]/).pop();
                        this.state.files.push({
                            path: filePath,
                            name: name,
                            status: 'pending'
                        });
                    }
                });
                this.renderFileList();
                this.updateButtonStates();
            }
        } catch (error) {
            console.error('Error selecting files:', error);
            ConsoleManager.log(`Error selecting files: ${error.message}`, 'error');
        }
    },

    /**
     * Remove selected files from the list
     */
    removeSelected() {
        const checkboxes = document.querySelectorAll('#translatorFileList input[type="checkbox"]:checked');
        const pathsToRemove = new Set();
        checkboxes.forEach(cb => pathsToRemove.add(cb.dataset.path));
        this.state.files = this.state.files.filter(f => !pathsToRemove.has(f.path));
        this.renderFileList();
        this.updateButtonStates();
    },

    /**
     * Clear all files from the list
     */
    clearFiles() {
        this.state.files = [];
        this.renderFileList();
        this.updateButtonStates();
    },

    /**
     * Render the file list
     */
    renderFileList() {
        const container = document.getElementById('translatorFileList');
        if (!container) return;

        if (this.state.files.length === 0) {
            container.innerHTML = `
                <div class="translator-empty-state">
                    <span class="empty-icon">📄</span>
                    <p>No SRT files selected</p>
                    <p class="empty-hint">Click "Add Files" to select subtitle files for translation</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.state.files.map((file, index) => `
            <div class="translator-file-item" data-index="${index}">
                <input type="checkbox" data-path="${file.path}">
                <span class="translator-file-name" title="${file.path}">${file.name}</span>
                <span class="translator-file-status ${file.status}">${this.formatStatus(file.status)}</span>
            </div>
        `).join('');

        // Bind click handlers for selection
        container.querySelectorAll('.translator-file-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (e.target.tagName !== 'INPUT') {
                    const checkbox = item.querySelector('input[type="checkbox"]');
                    checkbox.checked = !checkbox.checked;
                    this.updateButtonStates();
                }
            });
        });

        // Bind checkbox change handlers
        container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', () => this.updateButtonStates());
        });
    },

    /**
     * Format status for display
     */
    formatStatus(status) {
        const map = {
            pending: 'Pending',
            translating: 'Translating...',
            completed: 'Done',
            error: 'Error'
        };
        return map[status] || status;
    },

    /**
     * Update button states based on current state
     */
    updateButtonStates() {
        const hasFiles = this.state.files.length > 0;
        const hasSelected = document.querySelectorAll('#translatorFileList input[type="checkbox"]:checked').length > 0;

        document.getElementById('translatorRemoveSelected').disabled = !hasSelected || this.state.isRunning;
        document.getElementById('translatorClearFiles').disabled = !hasFiles || this.state.isRunning;
        document.getElementById('translatorStartBtn').disabled = !hasFiles || this.state.isRunning;
        document.getElementById('translatorCancelBtn').disabled = !this.state.isRunning;

        // Disable add files during translation
        document.getElementById('translatorAddFiles').disabled = this.state.isRunning;
    },

    /**
     * Update model dropdown options based on provider
     */
    updateModelOptions(provider) {
        const modelSelect = document.getElementById('translatorModel');
        if (!modelSelect) return;

        // Display labels for local models (show VRAM requirements)
        const localModelLabels = {
            'gemma-9b': 'gemma-9b (8GB)',
            'llama-8b': 'llama-8b (6GB)',
            'llama-3b': 'llama-3b (3GB)',
            'auto': 'auto'
        };

        const models = this.providerModels[provider] || [];
        modelSelect.innerHTML = '<option value="">Default</option>' +
            models.map(m => {
                const label = (provider === 'local') ? (localModelLabels[m] || m) : m;
                return `<option value="${m}">${label}</option>`;
            }).join('');
    },

    /**
     * Update API key status indicator
     */
    async updateApiKeyStatus(provider) {
        const statusEl = document.getElementById('translatorApiStatus');
        if (!statusEl) return;

        try {
            const result = await pywebview.api.get_translation_providers();
            if (result.success) {
                const providerInfo = result.providers.find(p => p.name === provider);
                if (providerInfo) {
                    if (providerInfo.has_api_key) {
                        statusEl.textContent = 'Configured';
                        statusEl.className = 'api-status configured';
                    } else {
                        statusEl.textContent = 'Not Set';
                        statusEl.className = 'api-status missing';
                    }
                }
            }
        } catch (error) {
            statusEl.textContent = 'Unknown';
            statusEl.className = 'api-status';
        }
    },

    /**
     * Test provider connection
     */
    async testConnection() {
        const provider = document.getElementById('translatorProvider')?.value;
        const apiKey = document.getElementById('translatorApiKey')?.value || null;

        const testBtn = document.getElementById('translatorTestConnection');
        const originalText = testBtn.textContent;
        testBtn.textContent = 'Testing...';
        testBtn.disabled = true;

        try {
            const result = await pywebview.api.test_provider_connection(provider, apiKey);
            const statusEl = document.getElementById('translatorApiStatus');

            if (result.success) {
                ConsoleManager.log(`Connection to ${provider} successful`, 'success');
                if (statusEl) {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'api-status configured';
                }
            } else {
                ConsoleManager.log(`Connection to ${provider} failed: ${result.error}`, 'error');
                if (statusEl) {
                    statusEl.textContent = 'Failed';
                    statusEl.className = 'api-status error';
                }
            }
        } catch (error) {
            ConsoleManager.log(`Connection test error: ${error.message}`, 'error');
        } finally {
            testBtn.textContent = originalText;
            testBtn.disabled = false;
        }
    },

    /**
     * Collect translation options from form
     */
    collectOptions() {
        return {
            input_files: this.state.files.map(f => f.path),
            provider: document.getElementById('translatorProvider')?.value || 'deepseek',
            model: document.getElementById('translatorModel')?.value ||
                   document.getElementById('translatorCustomModel')?.value || null,
            source_language: document.getElementById('translatorSourceLang')?.value || 'japanese',
            target_language: document.getElementById('translatorTargetLang')?.value || 'english',
            tone: document.getElementById('translatorTone')?.value || 'standard',
            movie_title: document.getElementById('translatorMovieTitle')?.value || null,
            actress: document.getElementById('translatorActress')?.value || null,
            plot: document.getElementById('translatorPlot')?.value || null,
            scene_threshold: parseFloat(document.getElementById('translatorSceneThreshold')?.value) || 60,
            max_batch_size: parseInt(document.getElementById('translatorMaxBatchSize')?.value) || 30,
            rate_limit: document.getElementById('translatorRateLimit')?.value ?
                       parseInt(document.getElementById('translatorRateLimit').value) : null,
            max_retries: parseInt(document.getElementById('translatorMaxRetries')?.value) || 3,
            endpoint: document.getElementById('translatorCustomEndpoint')?.value || null,
            api_key: document.getElementById('translatorApiKey')?.value || null
        };
    },

    /**
     * Start translation
     */
    async startTranslation() {
        if (this.state.files.length === 0) {
            ErrorHandler.show('No Files', 'Please add SRT files to translate.');
            return;
        }

        const options = this.collectOptions();

        try {
            this.state.isRunning = true;
            this.updateButtonStates();
            this.setProgress(0);
            this.setStatus('Starting...');

            // Mark all files as pending
            this.state.files.forEach(f => f.status = 'pending');
            this.renderFileList();

            const result = await pywebview.api.start_translation(options);

            if (result.success) {
                ConsoleManager.log('Translation started', 'info');
                this.startStatusPolling();
            } else {
                throw new Error(result.error || 'Failed to start translation');
            }
        } catch (error) {
            ConsoleManager.log(`Translation error: ${error.message}`, 'error');
            ErrorHandler.show('Translation Error', error.message);
            this.state.isRunning = false;
            this.updateButtonStates();
            this.setStatus('Error');
        }
    },

    /**
     * Cancel translation
     */
    async cancelTranslation() {
        try {
            await pywebview.api.cancel_translation();
            ConsoleManager.log('Translation cancelled', 'warning');
            this.stopStatusPolling();
            this.state.isRunning = false;
            this.updateButtonStates();
            this.setStatus('Cancelled');
        } catch (error) {
            console.error('Error cancelling translation:', error);
        }
    },

    /**
     * Start polling for translation status
     */
    startStatusPolling() {
        this.statusInterval = setInterval(async () => {
            try {
                const status = await pywebview.api.get_translation_status();

                // Update progress
                if (status.progress !== undefined) {
                    this.setProgress(status.progress);
                }

                // Fetch logs
                this.fetchLogs();

                if (status.status === 'completed') {
                    this.stopStatusPolling();
                    this.state.isRunning = false;
                    this.state.files.forEach(f => f.status = 'completed');
                    this.renderFileList();
                    this.updateButtonStates();
                    this.setProgress(100);
                    this.setStatus('Completed');
                    ConsoleManager.log('Translation completed successfully', 'success');
                    ErrorHandler.showSuccess('Translation Complete', 'All files have been translated.');
                } else if (status.status === 'error') {
                    this.stopStatusPolling();
                    this.state.isRunning = false;
                    this.updateButtonStates();
                    this.setStatus(`Error: ${status.error || 'Unknown'}`);
                    ConsoleManager.log(`Translation error: ${status.error}`, 'error');
                } else if (status.status === 'cancelled') {
                    this.stopStatusPolling();
                    this.state.isRunning = false;
                    this.updateButtonStates();
                    this.setStatus('Cancelled');
                }
            } catch (error) {
                console.error('Status poll error:', error);
            }
        }, 1000);
    },

    /**
     * Stop status polling
     */
    stopStatusPolling() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    },

    /**
     * Fetch and display translation logs
     */
    async fetchLogs() {
        try {
            const logs = await pywebview.api.get_translation_logs();
            if (logs && logs.length > 0) {
                logs.forEach(line => {
                    const cleanLine = line.replace(/\n$/, '');
                    if (cleanLine) {
                        ConsoleManager.appendRaw(cleanLine);
                    }
                });
            }
        } catch (error) {
            console.error('Error fetching logs:', error);
        }
    },

    /**
     * Set progress bar value
     */
    setProgress(percent) {
        this.state.progress = percent;
        const fill = document.getElementById('translatorProgressFill');
        if (fill) {
            fill.style.width = `${percent}%`;
        }
    },

    /**
     * Set status label
     */
    setStatus(text) {
        const label = document.getElementById('translatorStatusLabel');
        if (label) {
            label.textContent = text;
        }
    }
};

// ============================================================
// Translate Integration Manager
// ============================================================
const TranslateIntegrationManager = {
    // State
    state: {
        enabled: false,
        provider: 'deepseek',
        target: 'english',
        sourceLanguage: 'japanese',
        isTranslating: false,
        outputFiles: [],  // SRT files generated by transcription
        // Captured at process start time
        capturedSettings: null
    },

    init() {
        // Bind checkbox handlers for Transcription Mode
        const transcribeCheckbox = document.getElementById('translateAfterTranscription');
        const quickSettings = document.getElementById('translateQuickSettings');

        if (transcribeCheckbox && quickSettings) {
            transcribeCheckbox.addEventListener('change', () => {
                quickSettings.style.display = transcribeCheckbox.checked ? 'flex' : 'none';
                this.state.enabled = transcribeCheckbox.checked;
            });
        }

        // Bind checkbox handlers for Ensemble Mode
        const ensembleCheckbox = document.getElementById('ensembleTranslateAfter');
        const ensembleSettings = document.getElementById('ensembleTranslateSettings');

        if (ensembleCheckbox && ensembleSettings) {
            ensembleCheckbox.addEventListener('change', () => {
                ensembleSettings.style.display = ensembleCheckbox.checked ? 'flex' : 'none';
            });
        }

        // Bind provider/target selects for Transcription Mode
        const quickProvider = document.getElementById('quickTranslateProvider');
        const quickTarget = document.getElementById('quickTranslateTarget');

        if (quickProvider) {
            quickProvider.addEventListener('change', () => {
                this.state.provider = quickProvider.value;
            });
        }
        if (quickTarget) {
            quickTarget.addEventListener('change', () => {
                this.state.target = quickTarget.value;
            });
        }

        // Bind provider/target selects for Ensemble Mode
        const ensembleProvider = document.getElementById('ensembleTranslateProvider');
        const ensembleTarget = document.getElementById('ensembleTranslateTarget');

        if (ensembleProvider) {
            ensembleProvider.addEventListener('change', () => {
                // Sync with transcription mode if needed
            });
        }
        if (ensembleTarget) {
            ensembleTarget.addEventListener('change', () => {
                // Sync with transcription mode if needed
            });
        }

        console.log('TranslateIntegrationManager initialized');
    },

    /**
     * Capture translation settings at process start time.
     * Should be called when transcription/ensemble process starts.
     */
    captureSettingsOnStart() {
        const enabled = this.isEnabled();
        if (enabled) {
            this.state.capturedSettings = {
                enabled: true,
                ...this.getSettings(),
                sourceLanguage: document.getElementById('source-language')?.value || 'japanese'
            };
            console.log('Translation settings captured:', this.state.capturedSettings);
        } else {
            this.state.capturedSettings = null;
        }
    },

    /**
     * Check if translation was enabled when process started
     */
    wasEnabledOnStart() {
        return this.state.capturedSettings?.enabled || false;
    },

    /**
     * Check if translation is enabled for the current mode
     */
    isEnabled() {
        const activeTab = document.querySelector('.tab-button.active');
        if (!activeTab) return false;

        const tabId = activeTab.dataset.tab;

        if (tabId === 'tab1' || tabId === 'tab2') {
            // Transcription Mode or Advanced Options - use main checkbox
            const checkbox = document.getElementById('translateAfterTranscription');
            return checkbox && checkbox.checked;
        } else if (tabId === 'tab3') {
            // Ensemble Mode
            const checkbox = document.getElementById('ensembleTranslateAfter');
            return checkbox && checkbox.checked;
        }
        return false;
    },

    /**
     * Get translation settings for the current mode
     */
    getSettings() {
        const activeTab = document.querySelector('.tab-button.active');
        const tabId = activeTab ? activeTab.dataset.tab : 'tab1';

        if (tabId === 'tab3') {
            // Ensemble Mode settings - use TranslationSettingsModal
            const fullSettings = TranslationSettingsModal.getFullSettings();
            return {
                provider: fullSettings.provider,
                model: fullSettings.model,
                modelOverride: fullSettings.modelOverride,  // Custom model path/name override
                target: fullSettings.targetLang,
                tone: fullSettings.tone,
                apiKey: fullSettings.apiKey,
                movieTitle: fullSettings.movieTitle,
                actress: fullSettings.actress,
                plot: fullSettings.plot,
                sceneThreshold: fullSettings.sceneThreshold,
                maxBatchSize: fullSettings.maxBatchSize,
                temperature: fullSettings.temperature,
                topP: fullSettings.topP,
                customEndpoint: fullSettings.customEndpoint
            };
        } else {
            // Transcription Mode settings
            return {
                provider: document.getElementById('quickTranslateProvider')?.value || 'local',
                target: document.getElementById('quickTranslateTarget')?.value || 'english'
            };
        }
    },

    /**
     * Called when transcription completes successfully
     * @param {Array<string>} outputFiles - List of generated SRT file paths
     */
    async onTranscriptionComplete(outputFiles) {
        // Use captured settings from when process started
        const settings = this.state.capturedSettings;

        if (!settings || !settings.enabled || !outputFiles || outputFiles.length === 0) {
            this.state.capturedSettings = null;
            return;
        }

        ConsoleManager.log('Starting post-transcription translation...', 'info');
        ConsoleManager.log(`Provider: ${settings.provider}, Target: ${settings.target}`, 'info');

        this.state.isTranslating = true;
        this.state.outputFiles = outputFiles;

        try {
            // Start translation via API - use correct key names matching api.py
            const result = await pywebview.api.start_translation({
                inputs: outputFiles,              // API expects 'inputs' not 'input_files'
                provider: settings.provider,
                target: settings.target,          // API expects 'target' not 'target_language'
                model: settings.model || '',
                api_key: settings.apiKey || '',
                tone: settings.tone || 'standard',
                movie_title: settings.movieTitle || '',
                actress: settings.actress || '',
                movie_plot: settings.plot || '',
                endpoint: settings.customEndpoint || ''
            });

            if (result.success) {
                ConsoleManager.log('Translation started', 'success');
                // Start polling for translation status
                this.startStatusPolling();
            } else {
                ConsoleManager.log(`Translation failed to start: ${result.error}`, 'error');
                this.state.isTranslating = false;
            }
        } catch (error) {
            ConsoleManager.log(`Translation error: ${error.message}`, 'error');
            this.state.isTranslating = false;
        }

        // Clear captured settings
        this.state.capturedSettings = null;
    },

    /**
     * Poll translation status
     */
    startStatusPolling() {
        this.statusInterval = setInterval(async () => {
            try {
                const status = await pywebview.api.get_translation_status();

                // Fetch and display translation logs
                this.fetchLogs();

                if (status.status === 'completed') {
                    this.stopStatusPolling();
                    ConsoleManager.log('Translation completed successfully', 'success');
                    ErrorHandler.showSuccess('Translation Complete', 'Subtitles have been translated.');
                    this.state.isTranslating = false;
                } else if (status.status === 'error') {
                    this.stopStatusPolling();
                    ConsoleManager.log(`Translation error: ${status.error || 'Unknown error'}`, 'error');
                    this.state.isTranslating = false;
                } else if (status.status === 'cancelled') {
                    this.stopStatusPolling();
                    ConsoleManager.log('Translation cancelled', 'warning');
                    this.state.isTranslating = false;
                }
                // Continue polling if still running
            } catch (error) {
                console.error('Translation status poll error:', error);
            }
        }, 1000);
    },

    /**
     * Stop polling for translation status
     */
    stopStatusPolling() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    },

    /**
     * Fetch translation logs
     */
    async fetchLogs() {
        try {
            const logs = await pywebview.api.get_translation_logs();
            if (logs && logs.length > 0) {
                logs.forEach(line => {
                    ConsoleManager.appendRaw(line);
                });
            }
        } catch (error) {
            console.error('Error fetching translation logs:', error);
        }
    }
};

// ============================================================
// Translation Settings Modal Manager
// ============================================================
const TranslationSettingsModal = {
    // Provider model options (shared with inline dropdown)
    providerModels: {
        local: ['gemma-9b', 'llama-8b', 'llama-3b', 'auto'],
        deepseek: ['deepseek-chat', 'deepseek-coder'],
        gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
        claude: ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
        gpt: ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
        openrouter: ['deepseek/deepseek-chat', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o'],
        glm: ['glm-4-flash', 'glm-4', 'glm-4-plus'],
        groq: ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768']
    },

    // Settings state
    settings: {
        apiKey: '',
        targetLang: 'english',
        tone: 'standard',
        movieTitle: '',
        actress: '',
        plot: '',
        sceneThreshold: 60,
        maxBatchSize: 30,
        temperature: 0.5,
        topP: 0.9,
        customEndpoint: ''
    },

    init() {
        // Modal open/close handlers
        document.getElementById('ensembleTranslateSettingsBtn')?.addEventListener('click', () => this.open());
        document.getElementById('translationSettingsClose')?.addEventListener('click', () => this.close());
        document.getElementById('translationSettingsCancel')?.addEventListener('click', () => this.close());
        document.getElementById('translationSettingsSave')?.addEventListener('click', () => this.save());

        // Close on overlay click
        document.getElementById('translationSettingsModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'translationSettingsModal') this.close();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen()) this.close();
        });

        // Test connection button
        document.getElementById('translationTestConnection')?.addEventListener('click', () => this.testConnection());

        // Provider change handler for inline dropdown
        document.getElementById('ensembleTranslateProvider')?.addEventListener('change', (e) => {
            this.updateModelOptions(e.target.value);
        });

        // Initialize model options for default provider (Local LLM)
        this.updateModelOptions('local');

        // Set default model selection to gemma-9b (8GB VRAM)
        const modelSelect = document.getElementById('ensembleTranslateModel');
        if (modelSelect) {
            // Will be populated by updateModelOptions, then set default
            setTimeout(() => {
                if (modelSelect.querySelector('option[value="gemma-9b"]')) {
                    modelSelect.value = 'gemma-9b';
                }
            }, 0);
        }

        // Load saved settings
        this.loadSettings();

        console.log('TranslationSettingsModal initialized');
    },

    isOpen() {
        const modal = document.getElementById('translationSettingsModal');
        return modal && modal.classList.contains('active');
    },

    open() {
        const modal = document.getElementById('translationSettingsModal');
        if (modal) {
            // Populate form with current settings
            this.populateForm();

            // Show multi-file warning if applicable
            this.updateMultiFileWarning();

            modal.classList.add('active');
        }
    },

    close() {
        const modal = document.getElementById('translationSettingsModal');
        if (modal) {
            modal.classList.remove('active');
        }
    },

    save() {
        // Save settings from form
        this.settings.apiKey = document.getElementById('translationApiKey')?.value || '';
        this.settings.targetLang = document.getElementById('translationTargetLang')?.value || 'english';
        this.settings.tone = document.getElementById('translationTone')?.value || 'standard';
        this.settings.movieTitle = document.getElementById('translationMovieTitle')?.value || '';
        this.settings.actress = document.getElementById('translationActress')?.value || '';
        this.settings.plot = document.getElementById('translationPlot')?.value || '';
        this.settings.sceneThreshold = parseInt(document.getElementById('translationSceneThreshold')?.value) || 60;
        this.settings.maxBatchSize = parseInt(document.getElementById('translationMaxBatchSize')?.value) || 30;
        this.settings.temperature = parseFloat(document.getElementById('translationTemperature')?.value) || 0.5;
        this.settings.topP = parseFloat(document.getElementById('translationTopP')?.value) || 0.9;
        this.settings.customEndpoint = document.getElementById('translationCustomEndpoint')?.value || '';

        // Persist to local storage
        try {
            localStorage.setItem('whisperjav_translation_settings', JSON.stringify(this.settings));
        } catch (e) {
            console.warn('Could not save translation settings to localStorage:', e);
        }

        this.close();
        ConsoleManager.log('Translation settings saved', 'success');
    },

    loadSettings() {
        try {
            const saved = localStorage.getItem('whisperjav_translation_settings');
            if (saved) {
                const parsed = JSON.parse(saved);
                this.settings = { ...this.settings, ...parsed };
            }
        } catch (e) {
            console.warn('Could not load translation settings from localStorage:', e);
        }
    },

    populateForm() {
        document.getElementById('translationApiKey').value = this.settings.apiKey;
        document.getElementById('translationTargetLang').value = this.settings.targetLang;
        document.getElementById('translationTone').value = this.settings.tone;
        document.getElementById('translationMovieTitle').value = this.settings.movieTitle;
        document.getElementById('translationActress').value = this.settings.actress;
        document.getElementById('translationPlot').value = this.settings.plot;
        document.getElementById('translationSceneThreshold').value = this.settings.sceneThreshold;
        document.getElementById('translationMaxBatchSize').value = this.settings.maxBatchSize;
        document.getElementById('translationTemperature').value = this.settings.temperature;
        document.getElementById('translationTopP').value = this.settings.topP;
        document.getElementById('translationCustomEndpoint').value = this.settings.customEndpoint;
    },

    updateMultiFileWarning() {
        const warning = document.getElementById('multiFileWarning');
        if (warning) {
            // Show warning if more than 1 file is selected
            const fileCount = AppState.selectedFiles?.length || 0;
            warning.style.display = fileCount > 1 ? 'flex' : 'none';
        }
    },

    updateModelOptions(provider) {
        const modelSelect = document.getElementById('ensembleTranslateModel');
        if (!modelSelect) return;

        // Display labels for local models (show VRAM requirements)
        const localModelLabels = {
            'gemma-9b': 'gemma-9b (8GB)',
            'llama-8b': 'llama-8b (6GB)',
            'llama-3b': 'llama-3b (3GB)',
            'auto': 'auto'
        };

        const models = this.providerModels[provider] || [];
        modelSelect.innerHTML = '<option value="">Default</option>' +
            models.map(m => {
                const label = (provider === 'local') ? (localModelLabels[m] || m) : m;
                return `<option value="${m}">${label}</option>`;
            }).join('');

        // Set default model for local provider to gemma-9b
        if (provider === 'local' && models.includes('gemma-9b')) {
            modelSelect.value = 'gemma-9b';
        }
    },

    async testConnection() {
        const btn = document.getElementById('translationTestConnection');
        const statusEl = document.getElementById('translationApiStatus');
        const provider = document.getElementById('ensembleTranslateProvider')?.value || 'local';
        const apiKey = document.getElementById('translationApiKey')?.value || '';

        // Local LLM doesn't need API key testing
        if (provider === 'local') {
            if (statusEl) {
                statusEl.textContent = 'Local (No API)';
                statusEl.className = 'api-status connected';
            }
            ConsoleManager.log('Local LLM provider selected - no API key required', 'info');
            return;
        }

        if (btn) btn.disabled = true;
        if (statusEl) {
            statusEl.textContent = 'Testing...';
            statusEl.className = 'api-status testing';
        }

        try {
            const result = await pywebview.api.test_provider_connection(provider, apiKey);
            if (statusEl) {
                if (result.success) {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'api-status connected';
                    ConsoleManager.log(`API connection successful: ${provider}`, 'success');
                } else {
                    statusEl.textContent = 'Failed';
                    statusEl.className = 'api-status error';
                    ConsoleManager.log(`API connection failed: ${result.error || 'Unknown error'}`, 'error');
                }
            }
        } catch (error) {
            if (statusEl) {
                statusEl.textContent = 'Error';
                statusEl.className = 'api-status error';
            }
            ConsoleManager.log(`API connection error: ${error.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    },

    /**
     * Get all translation settings for processing
     */
    getFullSettings() {
        const provider = document.getElementById('ensembleTranslateProvider')?.value || 'local';
        const model = document.getElementById('ensembleTranslateModel')?.value || '';
        const modelOverride = document.getElementById('ensembleTranslateModelOverride')?.value?.trim() || '';

        return {
            provider: provider,
            model: model,
            modelOverride: modelOverride,  // Custom model path/name override
            apiKey: this.settings.apiKey,
            targetLang: this.settings.targetLang,
            tone: this.settings.tone,
            movieTitle: this.settings.movieTitle,
            actress: this.settings.actress,
            plot: this.settings.plot,
            sceneThreshold: this.settings.sceneThreshold,
            maxBatchSize: this.settings.maxBatchSize,
            temperature: this.settings.temperature,
            topP: this.settings.topP,
            customEndpoint: this.settings.customEndpoint
        };
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
    ModeManager.init();
    FileListManager.init();
    FormManager.init();
    ConsoleManager.init();
    ProgressManager.init();
    DirectoryControls.init();
    RunControls.init();
    KeyboardShortcuts.init();
    ThemeManager.init();
    EnsembleManager.init();
    UpdateManager.init();
    UpdateCheckManager.init();
    TranslatorManager.init();
    TranslateIntegrationManager.init();
    TranslationSettingsModal.init();

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

    // Update speech segmenter options based on backend availability
    EnsembleManager.updateSegmenterAvailability();

    // Update speech enhancer options based on backend availability
    EnsembleManager.updateEnhancerAvailability();
});
