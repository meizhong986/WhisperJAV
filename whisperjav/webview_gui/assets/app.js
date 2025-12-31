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

            // Scene detection method (optional)
            scene_detection_method: document.getElementById('sceneDetectionMethod').value,

            // Model override (conditional)
            model_override: modelOverrideEnabled
                ? document.getElementById('modelSelection').value
                : '',

            // Async processing (conditional)
            async_processing: asyncProcessingEnabled,

            // Speech segmenter selection (replaces no_vad checkbox)
            speech_segmenter: document.getElementById('speechSegmenter').value,

            // Other options
            credit: document.getElementById('openingCredit').value.trim(),

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
// Two-Pass Ensemble Manager
// ============================================================
const EnsembleManager = {
    // State - Full Configuration Snapshot approach
    state: {
        pass1: {
            pipeline: 'balanced',
            sensitivity: 'aggressive',
            sceneDetector: 'auditok',
            speechEnhancer: 'none',
            speechSegmenter: 'silero-v4.0',  // Explicit Silero V4.0 default
            model: 'large-v2',
            customized: false,
            params: null,  // null = use defaults, object = full custom config
            isTransformers: false,  // Track if using Transformers pipeline
            dspEffects: ['loudnorm']  // Default FFmpeg DSP effects
        },
        pass2: {
            enabled: false,
            pipeline: 'transformers',
            sensitivity: 'aggressive',
            sceneDetector: 'none',
            speechEnhancer: 'none',
            speechSegmenter: 'none',
            model: 'kotoba-tech/kotoba-whisper-bilingual-v1.0',  // Kotoba Bilingual for mixed ja/en
            customized: false,
            params: null,
            isTransformers: true,  // Default Pass 2 is Transformers
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

        // Update isTransformers flags based on synced pipeline values
        this.state.pass1.isTransformers = this.state.pass1.pipeline === 'transformers';
        this.state.pass2.isTransformers = this.state.pass2.pipeline === 'transformers';

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
        const wasTransformers = passState.isTransformers;

        if (passState.customized) {
            // Warn user that custom params will be reset
            if (confirm(`You have custom parameters for ${passKey === 'pass1' ? 'Pass 1' : 'Pass 2'}. Changing the pipeline will reset them. Continue?`)) {
                passState.pipeline = newValue;
                passState.customized = false;
                passState.params = null;
                passState.isTransformers = isTransformers;
                this.updateBadges();
                this.updateRowGreyingState(passKey);
                // Swap model options if pipeline type changed
                if (wasTransformers !== isTransformers) {
                    this.swapModelOptions(passKey, isTransformers);
                }
            } else {
                // Revert selection
                selectElement.value = passState.pipeline;
            }
        } else {
            passState.pipeline = newValue;
            passState.isTransformers = isTransformers;
            this.updateRowGreyingState(passKey);
            // Swap model options if pipeline type changed
            if (wasTransformers !== isTransformers) {
                this.swapModelOptions(passKey, isTransformers);
            }
        }
    },

    // Swap model dropdown options based on pipeline type
    swapModelOptions(passKey, isTransformers) {
        const modelSelect = document.getElementById(`${passKey}-model`);
        const models = isTransformers ? this.transformersModels : this.legacyModels;

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

    // Update row greying based on pipeline type (Transformers disables Sensitivity + Speech Segmenter)
    updateRowGreyingState(passKey) {
        const passState = this.state[passKey];
        const sensitivitySelect = document.getElementById(`${passKey}-sensitivity`);
        const segmenterSelect = document.getElementById(`${passKey}-segmenter`);

        // Check if pass2 is disabled (controls should be disabled regardless of pipeline)
        const isPass2Disabled = passKey === 'pass2' && !this.state.pass2.enabled;

        if (passState.isTransformers) {
            // Disable sensitivity and speech segmenter for Transformers
            // Note: Transformers uses HuggingFace internal chunking, not external VAD
            // Speech segmentation support planned for v1.8.0
            sensitivitySelect.disabled = true;
            sensitivitySelect.title = 'Sensitivity not applicable for Transformers mode';
            segmenterSelect.disabled = true;
            segmenterSelect.title = 'Transformers uses HF internal chunking (segmentation planned for v1.8.0)';

            // Set to 'none' for visual clarity
            if (sensitivitySelect.querySelector('option[value="none"]')) {
                sensitivitySelect.value = 'none';
                passState.sensitivity = 'none';
            }
            if (segmenterSelect.querySelector('option[value="none"]')) {
                segmenterSelect.value = 'none';
                passState.speechSegmenter = 'none';
            }
        } else {
            // Re-enable sensitivity and segmenter (unless pass2 is disabled)
            sensitivitySelect.disabled = isPass2Disabled;
            sensitivitySelect.title = '';
            segmenterSelect.disabled = isPass2Disabled;
            segmenterSelect.title = '';
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

        // Route to Transformers-specific handler if applicable
        if (passState.isTransformers) {
            await this.openTransformersCustomize(passKey);
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

        const config = {
            inputs: AppState.selectedFiles,
            output_dir: document.getElementById('outputDir').value,
            pass1: {
                pipeline: this.state.pass1.pipeline,
                sensitivity: this.state.pass1.isTransformers ? null : this.state.pass1.sensitivity,
                sceneDetector: this.state.pass1.sceneDetector,
                speechEnhancer: this.state.pass1.speechEnhancer,
                dspEffects: this.state.pass1.speechEnhancer === 'ffmpeg-dsp' ? this.state.pass1.dspEffects : null,
                speechSegmenter: this.state.pass1.isTransformers ? null : this.state.pass1.speechSegmenter,
                model: this.state.pass1.model,
                customized: this.state.pass1.customized,
                params: this.state.pass1.customized ? this.state.pass1.params : null,
                isTransformers: this.state.pass1.isTransformers
            },
            pass2: {
                enabled: this.state.pass2.enabled,
                pipeline: this.state.pass2.pipeline,
                sensitivity: this.state.pass2.isTransformers ? null : this.state.pass2.sensitivity,
                sceneDetector: this.state.pass2.sceneDetector,
                speechEnhancer: this.state.pass2.speechEnhancer,
                dspEffects: this.state.pass2.speechEnhancer === 'ffmpeg-dsp' ? this.state.pass2.dspEffects : null,
                speechSegmenter: this.state.pass2.isTransformers ? null : this.state.pass2.speechSegmenter,
                model: this.state.pass2.model,
                customized: this.state.pass2.customized,
                params: this.state.pass2.customized ? this.state.pass2.params : null,
                isTransformers: this.state.pass2.isTransformers
            },
            merge_strategy: this.state.mergeStrategy,
            source_language: document.getElementById('source-language').value,
            subs_language: document.getElementById('language').value,
            debug: document.getElementById('debugLogging').checked,
            keep_temp: document.getElementById('keepTemp').checked,
            temp_dir: document.getElementById('tempDir').value.trim()
        };

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
