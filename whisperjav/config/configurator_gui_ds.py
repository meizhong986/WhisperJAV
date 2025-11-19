import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QRadioButton, QButtonGroup, QLabel, QScrollArea, QFormLayout, QLineEdit,
    QDoubleSpinBox, QCheckBox, QPushButton, QFileDialog, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class TranscriptionTuner:
    """Configuration resolver that works with in-memory config"""
    def __init__(self, config):
        self.config = config
    
    def resolve_params(self, pipeline_name, sensitivity, task):
        """Resolve configurations for given parameters"""
        if pipeline_name not in self.config["pipelines"]:
            raise ValueError(f"Unknown pipeline: '{pipeline_name}'")
        if sensitivity not in self.config["sensitivity_profiles"]:
            raise ValueError(f"Unknown sensitivity: '{sensitivity}'")
        
        pipeline_cfg = self.config["pipelines"][pipeline_name]
        sensitivity_profile = self.config["sensitivity_profiles"][sensitivity]
        
        # Determine model considering task-based overrides
        model_id = pipeline_cfg.get("model_overrides", {}).get(task, pipeline_cfg["workflow"]["model"])
        model_cfg = self.config["models"][model_id]
        provider = model_cfg["provider"]
        
        # Get all parameters
        params = {
            "decoder": self.config["parameter_sets"]["decoder_params"][sensitivity_profile["decoder"]],
            "vad": self.config["parameter_sets"]["vad_params"][sensitivity_profile["vad"]],
            "provider": self.config["parameter_sets"]["provider_specific_params"][provider][sensitivity_profile["provider_settings"]]
        }
        
        # Get feature configurations
        features = {}
        for feature, setting in pipeline_cfg["workflow"].get("features", {}).items():
            if setting and setting != "none":
                config_name = setting if isinstance(setting, str) else "default"
                if feature in self.config["feature_configs"] and config_name in self.config["feature_configs"][feature]:
                    features[feature] = self.config["feature_configs"][feature][config_name]
        
        return {
            "pipeline_name": pipeline_name,
            "sensitivity_name": sensitivity,
            "workflow": pipeline_cfg["workflow"],
            "model": model_cfg,
            "params": params,
            "features": features,
            "task": task,
            "language": self.config.get("defaults", {}).get("language", "ja")
        }

class ASRConfigurator(QMainWindow):
    def __init__(self, config_path=None):
        super().__init__()
        self.setWindowTitle("ASR Configurator v3")
        self.setGeometry(100, 100, 900, 800)
        
        # Load configuration
        self.config_path = config_path or Path("asr_config.v3.json")
        self.config = self.load_config()
        self.tuner = TranscriptionTuner(self.config)
        
        # Current selections
        self.selected_pipeline = "balanced"
        self.selected_sensitivity = "aggressive"
        self.selected_task = "transcribe"
        
        self.init_ui()
        self.update_resolved_settings()
        
    def load_config(self):
        """Load and parse the configuration file"""
        if not self.config_path.exists():
            QMessageBox.critical(self, "Error", f"Config file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # Strip comments
                config_text = "".join(line for line in f if not line.strip().startswith("//"))
                return json.loads(config_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")
            return {}

    def save_config(self):
        """Save configuration back to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            QMessageBox.information(self, "Success", "Configuration saved successfully!")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {str(e)}")
            return False

    def init_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create scroll area for the content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Section 1: Pipeline Selection
        pipeline_group = QGroupBox("1. Select Pipeline")
        pipeline_layout = QHBoxLayout(pipeline_group)
        self.pipeline_group = QButtonGroup(self)
        
        for pipeline in self.config["pipelines"]:
            btn = QRadioButton(pipeline.capitalize())
            btn.setChecked(pipeline == self.selected_pipeline)
            btn.toggled.connect(lambda checked, p=pipeline: self.on_pipeline_changed(p) if checked else None)
            pipeline_layout.addWidget(btn)
            self.pipeline_group.addButton(btn)
        
        # Section 2: Sensitivity Selection
        sensitivity_group = QGroupBox("2. Select Sensitivity")
        sensitivity_layout = QHBoxLayout(sensitivity_group)
        self.sensitivity_group = QButtonGroup(self)
        
        for sensitivity in self.config["sensitivity_profiles"]:
            btn = QRadioButton(sensitivity.capitalize())
            btn.setChecked(sensitivity == self.selected_sensitivity)
            btn.toggled.connect(lambda checked, s=sensitivity: self.on_sensitivity_changed(s) if checked else None)
            sensitivity_layout.addWidget(btn)
            self.sensitivity_group.addButton(btn)
        
        # Section 3: Task Selection
        task_group = QGroupBox("3. Select Task")
        task_layout = QHBoxLayout(task_group)
        self.task_group = QButtonGroup(self)
        
        tasks = ["transcribe", "translate"]
        for task in tasks:
            btn = QRadioButton(task.capitalize())
            btn.setChecked(task == self.selected_task)
            btn.toggled.connect(lambda checked, t=task: self.on_task_changed(t) if checked else None)
            task_layout.addWidget(btn)
            self.task_group.addButton(btn)
        
        # Section 4: Resolved Settings
        resolved_group = QGroupBox("4. Resolved Settings")
        resolved_layout = QVBoxLayout(resolved_group)
        
        # Header
        self.header_label = QLabel("")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        self.header_label.setFont(header_font)
        resolved_layout.addWidget(self.header_label)
        
        # Create a tab-like structure for sections
        self.section_container = QWidget()
        section_layout = QVBoxLayout(self.section_container)
        
        # Workflow section
        self.workflow_group = self.create_section_group("Workflow")
        section_layout.addWidget(self.workflow_group)
        
        # Model section
        self.model_group = self.create_section_group("Model")
        section_layout.addWidget(self.model_group)
        
        # Parameters section
        self.params_group = self.create_section_group("Params")
        section_layout.addWidget(self.params_group)
        
        # Add to resolved layout
        resolved_layout.addWidget(self.section_container)
        
        # Section 5: Parameter Sets
        param_sets_group = QGroupBox("Parameter Sets")
        param_sets_layout = QVBoxLayout(param_sets_group)
        
        # Feature configs
        feature_group = QGroupBox("Feature Configs")
        feature_layout = QFormLayout(feature_group)
        
        # Add sample controls (would be dynamically generated in a real app)
        feature_layout.addRow("Min speech duration ms:", QLineEdit("100"))
        
        # Provider specific params
        provider_group = QGroupBox("Provider Specific Params")
        provider_layout = QFormLayout(provider_group)
        
        # Add sample controls
        provider_layout.addRow("Provider:", QComboBox())
        provider_layout.addRow("No speech threshold:", QDoubleSpinBox())
        provider_layout.addRow("Logprob threshold:", QDoubleSpinBox())
        provider_layout.addRow("Logprob margin:", QDoubleSpinBox())
        provider_layout.addRow("Drop nonverbal vocals:", QCheckBox())
        provider_layout.addRow("Suppress blank:", QCheckBox())
        
        param_sets_layout.addWidget(feature_group)
        param_sets_layout.addWidget(provider_group)
        
        # Add all sections to content layout
        content_layout.addWidget(pipeline_group)
        content_layout.addWidget(sensitivity_group)
        content_layout.addWidget(task_group)
        content_layout.addWidget(resolved_group)
        content_layout.addWidget(param_sets_group)
        content_layout.addStretch()
        
        # Add file selection button
        file_btn = QPushButton("Select Config from Disk...")
        file_btn.clicked.connect(self.select_config_file)
        
        # Add save button
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_config)
        
        # Add button container
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addWidget(file_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        
        # Set up main layout
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(btn_container)
        self.setCentralWidget(main_widget)

    def create_section_group(self, title):
        """Create a collapsible section group"""
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)
        group.setStyleSheet("QGroupBox::title { font-weight: bold; }")
        return group

    def on_pipeline_changed(self, pipeline):
        """Handle pipeline selection change"""
        self.selected_pipeline = pipeline
        self.update_resolved_settings()

    def on_sensitivity_changed(self, sensitivity):
        """Handle sensitivity selection change"""
        self.selected_sensitivity = sensitivity
        self.update_resolved_settings()

    def on_task_changed(self, task):
        """Handle task selection change"""
        self.selected_task = task
        self.update_resolved_settings()

    def update_resolved_settings(self):
        """Update the resolved settings display"""
        try:
            # Resolve configuration
            resolved = self.tuner.resolve_params(
                self.selected_pipeline,
                self.selected_sensitivity,
                self.selected_task
            )
            
            # Update header
            self.header_label.setText(
                f"Pipeline name: {resolved['pipeline_name']}\n"
                f"Sensitivity name: {resolved['sensitivity_name']}"
            )
            
            # Update workflow section
            self.update_section(
                self.workflow_group,
                resolved["workflow"],
                ["model", "vad", "backend"],
                "Workflow"
            )
            
            # Add feature information if available
            if resolved["features"]:
                features_text = "\n".join([f"  - {k}: {v}" for k, v in resolved["features"].items()])
                self.add_to_section(self.workflow_group, "Features:", features_text)
            
            # Update model section
            self.update_section(
                self.model_group,
                resolved["model"],
                ["provider", "model_name", "device", "compute_type"],
                "Model"
            )
            
            # Add supported tasks if available
            if "supported_tasks" in resolved["model"]:
                tasks = ", ".join(resolved["model"]["supported_tasks"])
                self.add_to_section(self.model_group, "Supported tasks:", tasks)
            
            # Update parameters section
            self.params_group.setTitle("Params")
            params_layout = QVBoxLayout()
            self.params_group.setLayout(params_layout)
            
            # Decoder parameters
            decoder_group = QGroupBox("Decoder")
            decoder_layout = QFormLayout(decoder_group)
            self.add_params_form(decoder_layout, resolved["params"]["decoder"])
            params_layout.addWidget(decoder_group)
            
            # VAD parameters
            vad_group = QGroupBox("VAD")
            vad_layout = QFormLayout(vad_group)
            self.add_params_form(vad_layout, resolved["params"]["vad"])
            params_layout.addWidget(vad_group)
            
            # Provider parameters
            provider_group = QGroupBox("Provider")
            provider_layout = QFormLayout(provider_group)
            self.add_params_form(provider_layout, resolved["params"]["provider"])
            params_layout.addWidget(provider_group)
            
        except Exception as e:
            QMessageBox.warning(self, "Configuration Error", str(e))

    def update_section(self, group, data, keys, title):
        """Update a section with the given data"""
        group.setTitle(title)
        layout = QVBoxLayout()
        
        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    value = ", ".join(value)
                layout.addWidget(QLabel(f"• {key.capitalize()}: {value}"))
        
        group.setLayout(layout)

    def add_to_section(self, group, label, text):
        """Add additional information to a section"""
        layout = group.layout()
        if layout:
            layout.addWidget(QLabel(label))
            layout.addWidget(QLabel(text))

    def add_params_form(self, layout, params):
        """Add parameters to a form layout"""
        for key, value in params.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            layout.addRow(f"{key.replace('_', ' ').title()}:", QLabel(str(value)))

    def select_config_file(self):
        """Open file dialog to select a different config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration File",
            str(self.config_path.parent),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.config_path = Path(file_path)
            self.config = self.load_config()
            self.tuner = TranscriptionTuner(self.config)
            self.update_resolved_settings()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Use first argument as config path if provided
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    
    window = ASRConfigurator(config_path)
    window.show()
    sys.exit(app.exec_())