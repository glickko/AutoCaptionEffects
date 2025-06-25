document.addEventListener('DOMContentLoaded', function () {
    // Populate fonts dropdown and load settings when the window is ready
    window.addEventListener('pywebviewready', function() {
        pywebview.api.get_fonts().then(fonts => {
            const fontSelect = document.getElementById('font');
            fonts.forEach(fontName => {
                const option = document.createElement('option');
                option.value = fontName;
                option.textContent = fontName;
                fontSelect.appendChild(option);
            });
        });
        pywebview.api.load_settings().then(applySettings);
    });

    // Handle conditional UI visibility
    const presetSelect = document.getElementById('preset');
    presetSelect.addEventListener('change', updateEffectOptions);
    const strokeCheckbox = document.getElementById('stroke_enabled');
    strokeCheckbox.addEventListener('change', toggleStrokeOptions);
    
    // Set initial state after a short delay to ensure settings are loaded
    setTimeout(() => {
        updateEffectOptions();
        toggleStrokeOptions();
    }, 100);
});

// Tab Logic
function openTab(evt, tabName) {
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
    }
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Settings Logic
function getSettingsFromUI() {
    const settingIds = [
        'output_width', 'output_height', 'model_size', 'language', 'refine_timestamps', 
        'font', 'font_color', 'font_size_override', 'placement', 'stroke_enabled',
        'stroke_color', 'stroke_width', 'preset', 'highlight_color', 'highlight_opacity',
        'fadein_duration', 'popin_duration', 'highlight_style', 'highlight_padding', 
        'highlight_movement', 
        'split_by_punctuation', 'split_by_gap', 'gap_value', 'merge_gap_value',
        'max_chars', 
        'format_text', 'remove_punctuation', 'export_srt', 'censor_list'
    ];
    const settings = {};
    settingIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            if (el.type === 'checkbox') {
                settings[id] = el.checked;
            } else {
                settings[id] = el.value;
            }
        }
    });
    return settings;
}

function applySettings(settings) {
    if (!settings) return;
    for (const key in settings) {
        const el = document.getElementById(key);
        if (el) {
            if (el.type === 'checkbox') {
                el.checked = settings[key];
            } else {
                el.value = settings[key];
            }
        }
    }
    updateEffectOptions();
    toggleStrokeOptions();
}

function saveSettings() {
    const settings = getSettingsFromUI();
    pywebview.api.save_settings(settings).then(success => {
        if (success) {
            console.log("Settings saved!");
        } else {
            console.error("Failed to save settings.");
        }
    });
}

// Conditional UI Logic
function toggleStrokeOptions() {
    const container = document.getElementById('stroke-options-container');
    const isEnabled = document.getElementById('stroke_enabled').checked;
    if (isEnabled) {
        container.classList.remove('hidden');
    } else {
        container.classList.add('hidden');
    }
}

function updateEffectOptions() {
    const selectedPreset = document.getElementById('preset').value;
    const allOptions = document.querySelectorAll('.effect-option');

    allOptions.forEach(option => {
        const presets = option.getAttribute('data-preset').split(' ');
        if (presets.includes(selectedPreset)) {
            option.style.display = 'block'; 
        } else {
            option.style.display = 'none';
        }
    });
}

function generateCaptions() {
    const processButton = document.getElementById('process-button');
    processButton.disabled = true;
    updateProgress(0, "Preparing...");

    const params = getSettingsFromUI();
    params.input_file = document.getElementById('input_file').value;
    
    // Convert types from string where necessary
    params.output_width = parseInt(params.output_width);
    params.output_height = parseInt(params.output_height);
    params.stroke_width = parseInt(params.stroke_width);
    params.highlight_opacity = parseFloat(params.highlight_opacity);
    params.fadein_duration = parseFloat(params.fadein_duration);
    params.popin_duration = parseFloat(params.popin_duration);
    params.highlight_padding = parseInt(params.highlight_padding);
    params.max_chars = parseInt(params.max_chars);
    params.gap_value = parseFloat(params.gap_value);
    params.merge_gap_value = parseFloat(params.merge_gap_value);

    pywebview.api.process_media(params);
}

// Functions called by Python
function updateProgress(percentage, message) {
    const progressBar = document.getElementById('progress-bar');
    const progressLabel = document.getElementById('progress-label');
    const processButton = document.getElementById('process-button');

    progressBar.style.width = percentage + '%';
    progressLabel.textContent = message;

    if (percentage >= 100 || message.toLowerCase().includes("error")) {
        processButton.disabled = false;
         if (percentage >= 100) {
            progressLabel.textContent = "Completed!";
        }
    } else {
         processButton.disabled = true;
    }
}

function showError(message) {
    const errorBox = document.getElementById('error-box');
    const errorMessage = document.getElementById('error-message');
    
    errorMessage.textContent = `Error: ${message}`;
    errorBox.classList.remove('hidden');

    updateProgress(0, "Error Occurred");
}