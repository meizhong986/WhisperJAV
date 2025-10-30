# Aggressive Compactness - CSS Changes Summary

**Quick Reference Guide for Developers**

This document provides a concise, searchable reference of ALL CSS changes made during the aggressive compactness optimization.

---

## CSS Variables (`:root`)

```css
/* BEFORE */
--font-size-base: 13px;
--font-size-sm: 11px;
--font-size-lg: 15px;
--line-height: 1.5;

/* AFTER */
--font-size-base: 13px;      /* UNCHANGED */
--font-size-sm: 10px;        /* -1px */
--font-size-lg: 12px;        /* -3px */
--line-height: 1.4;          /* -0.1 */
```

---

## Header (`.app-header`)

```css
/* BEFORE */
.app-header {
    padding: 12px 16px 8px;
}

.app-header h1 {
    font-size: 18px;
    margin: 0;
    letter-spacing: -0.3px;
}

.app-header .subtitle {
    font-size: var(--font-size-sm);  /* 11px */
    opacity: 0.9;
    margin: 1px 0 0;
}

/* AFTER */
.app-header {
    padding: 4px 12px;           /* -8px vertical */
    min-height: 25px;            /* NEW */
}

.app-header h1 {
    font-size: 12px;             /* -6px */
    padding: 2px 0;              /* NEW */
    letter-spacing: -0.2px;      /* -0.1px */
}

.app-header .subtitle {
    display: none;               /* HIDDEN */
}
```

---

## Main & Footer

```css
/* BEFORE */
.app-main {
    padding: 10px 14px;
}

.app-footer {
    padding: 6px 16px;
    font-size: var(--font-size-sm);  /* 11px */
}

/* AFTER */
.app-main {
    padding: 8px 12px;           /* -2px vertical, -2px horizontal */
}

.app-footer {
    padding: 4px 12px;           /* -2px vertical, -4px horizontal */
    font-size: var(--font-size-sm);  /* Now 10px */
}
```

---

## Sections (`.section`)

```css
/* BEFORE */
.section {
    margin-bottom: 10px;
}

.section-header {
    padding: 8px 12px;
}

.section-header h2 {
    font-size: 15px;
    margin: 0;
}

.section-content {
    padding: 12px;
}

/* AFTER */
.section {
    margin-bottom: 8px;          /* -2px */
}

.section-header {
    padding: 4px 12px;           /* -4px vertical */
    min-height: 22px;            /* NEW */
}

.section-header h2 {
    font-size: 10px;             /* -5px */
    margin: 0;
    text-transform: uppercase;   /* NEW */
    letter-spacing: 0.5px;       /* NEW */
}

.section-content {
    padding: 10px;               /* -2px */
}
```

---

## Source Section - File List

```css
/* BEFORE */
.file-list-container {
    margin-bottom: 10px;
}

.file-list {
    min-height: 80px;
    max-height: 80px;
    padding: 8px;
}

.file-item {
    padding: 6px 10px;
    margin-bottom: 3px;
    gap: 8px;
}

/* AFTER */
.file-list-container {
    margin-bottom: 8px;          /* -2px */
}

.file-list {
    min-height: 70px;            /* -10px */
    max-height: 70px;            /* -10px */
    padding: 6px;                /* -2px */
}

.file-item {
    padding: 4px 8px;            /* -2px vertical, -2px horizontal */
    margin-bottom: 2px;          /* -1px */
    gap: 6px;                    /* -2px */
}
```

---

## Destination Section - Output Control

```css
/* BEFORE */
.output-control {
    gap: 8px;
}

.control-label {
    font-size: var(--font-size-base);  /* 13px */
}

.output-path {
    padding: 6px 10px;
    font-size: 12px;
}

/* AFTER */
.output-control {
    gap: 6px;                    /* -2px */
}

.control-label {
    font-size: 11px;             /* -2px */
}

.output-path {
    padding: 5px 8px;            /* -1px vertical, -2px horizontal */
    font-size: 11px;             /* -1px */
}
```

---

## Tabs Section

```css
/* BEFORE */
.tab-bar {
    padding: 0 16px;
}

.tab-button {
    padding: 8px 30px 6px;
    font-size: var(--font-size-base);  /* 13px */
    margin-right: 6px;
}

.tab-button.active {
    font-size: 14px;
    padding: 10px 30px 8px;
}

.tab-content-container {
    padding: 14px;
    min-height: 160px;
}

/* AFTER */
.tab-bar {
    padding: 0 12px;             /* -4px horizontal */
}

.tab-button {
    padding: 6px 24px 4px;       /* -2px top, -6px horizontal, -2px bottom */
    font-size: 11px;             /* -2px */
    margin-right: 4px;           /* -2px */
}

.tab-button.active {
    font-size: 12px;             /* -2px */
    padding: 8px 24px 6px;       /* -2px top, -6px horizontal, -2px bottom */
}

.tab-content-container {
    padding: 10px;               /* -4px */
    min-height: 140px;           /* -20px */
}
```

---

## Form Elements

```css
/* BEFORE */
.form-row {
    gap: 12px;
    margin-bottom: 12px;
}

.form-group {
    gap: 4px;
}

.form-label {
    font-size: 12px;
}

.form-input,
.form-select {
    padding: 6px 10px;
    font-size: var(--font-size-base);  /* 13px */
}

/* AFTER */
.form-row {
    gap: 10px;                   /* -2px */
    margin-bottom: 8px;          /* -4px */
}

.form-group {
    gap: 3px;                    /* -1px */
}

.form-label {
    font-size: 10px;             /* -2px */
}

.form-input,
.form-select {
    padding: 5px 8px;            /* -1px vertical, -2px horizontal */
    font-size: 12px;             /* -1px */
}
```

---

## Radio Buttons & Checkboxes

```css
/* BEFORE */
.radio-group {
    gap: 12px;
}

.radio-label {
    gap: 6px;
    font-size: var(--font-size-base);  /* 13px */
}

.radio-label input[type="radio"] {
    width: 16px;
    height: 16px;
}

.checkbox-label {
    gap: 8px;
    font-size: var(--font-size-base);  /* 13px */
}

.checkbox-label input[type="checkbox"] {
    width: 16px;
    height: 16px;
}

/* AFTER */
.radio-group {
    gap: 10px;                   /* -2px */
}

.radio-label {
    gap: 5px;                    /* -1px */
    font-size: 11px;             /* -2px */
}

.radio-label input[type="radio"] {
    width: 15px;                 /* -1px */
    height: 15px;                /* -1px */
}

.checkbox-label {
    gap: 6px;                    /* -2px */
    font-size: 11px;             /* -2px */
}

.checkbox-label input[type="checkbox"] {
    width: 15px;                 /* -1px */
    height: 15px;                /* -1px */
}
```

---

## Info Text (Help Text)

```css
/* BEFORE */
.info-row {
    margin-top: 6px;
}

.info-group {
    padding: 6px;
}

.info-text {
    font-size: 11px;
    line-height: 1.3;
}

/* AFTER */
.info-row {
    margin-top: 4px;             /* -2px */
}

.info-group {
    padding: 5px;                /* -1px */
}

.info-text {
    font-size: 10px;             /* -1px */
    line-height: 1.3;            /* UNCHANGED */
}
```

---

## Buttons

```css
/* BEFORE */
.btn {
    padding: 6px 12px;
    font-size: var(--font-size-base);  /* 13px */
    gap: 6px;
}

.btn-sm {
    padding: 5px 10px;
    font-size: var(--font-size-sm);  /* 11px */
}

.btn-lg {
    padding: 10px 20px;
    font-size: var(--font-size-lg);  /* 15px */
}

.button-group {
    gap: 6px;
}

/* AFTER */
.btn {
    padding: 5px 10px;           /* -1px vertical, -2px horizontal */
    font-size: 12px;             /* -1px */
    gap: 5px;                    /* -1px */
    min-height: 26px;            /* NEW */
}

.btn-sm {
    padding: 4px 8px;            /* -1px vertical, -2px horizontal */
    font-size: var(--font-size-sm);  /* Now 10px */
    min-height: 22px;            /* NEW */
}

.btn-lg {
    padding: 8px 16px;           /* -2px vertical, -4px horizontal */
    font-size: var(--font-size-lg);  /* Now 12px */
    min-height: 30px;            /* NEW */
}

.button-group {
    gap: 5px;                    /* -1px */
}
```

---

## Run Section - Progress & Controls

```css
/* BEFORE */
.run-section .section-content {
    padding: 12px;
}

.progress-container {
    gap: 10px;
    margin-bottom: 10px;
}

.progress-bar {
    height: 6px;
    border-radius: 3px;
}

.status-label {
    font-size: var(--font-size-base);  /* 13px */
    min-width: 70px;
}

/* AFTER */
.run-section .section-content {
    padding: 10px;               /* -2px */
}

.progress-container {
    gap: 8px;                    /* -2px */
    margin-bottom: 8px;          /* -2px */
}

.progress-bar {
    height: 5px;                 /* -1px */
    border-radius: 2.5px;        /* Proportional */
}

.status-label {
    font-size: 11px;             /* -2px */
    min-width: 60px;             /* -10px */
}
```

---

## Console Section

```css
/* BEFORE */
.console-output {
    font-size: 12px;
    padding: 10px;
    line-height: 1.5;
    min-height: 250px;
    max-height: none;
}

/* AFTER */
.console-output {
    font-size: 11px;             /* -1px */
    padding: 8px;                /* -2px */
    line-height: 1.45;           /* -0.05 */
    min-height: 280px;           /* +30px (INCREASED!) */
    max-height: none;            /* UNCHANGED */
}
```

---

## Responsive (Mobile)

```css
/* BEFORE */
@media (max-width: 900px) {
    .tab-button {
        padding: 6px 16px 4px;
    }

    .tab-button.active {
        padding: 8px 16px 6px;
    }
}

/* AFTER */
@media (max-width: 900px) {
    .tab-button {
        padding: 5px 14px 3px;   /* -1px top, -2px horizontal, -1px bottom */
        font-size: 10px;         /* NEW */
    }

    .tab-button.active {
        padding: 6px 14px 4px;   /* -2px top, -2px horizontal, -2px bottom */
        font-size: 11px;         /* NEW */
    }
}
```

---

## HTML Changes

### Header Subtitle Removed

```html
<!-- BEFORE -->
<header class="app-header">
    <h1>WhisperJAV</h1>
    <p class="subtitle">Simple Runner</p>
</header>

<!-- AFTER -->
<header class="app-header">
    <h1>WhisperJAV</h1>
</header>
```

---

## Search & Replace Patterns

If you need to find these changes in the CSS file, search for:

```
/* AGGRESSIVE: */
```

This comment appears before every changed line, making it easy to identify and modify/rollback specific changes.

---

## Quick Rollback Reference

### Restore Single Element

Example: Restore button font size only:

```css
.btn {
    font-size: 13px;  /* Change back from 12px */
}
```

### Restore Section Titles

```css
.section-header h2 {
    font-size: 15px;  /* Change back from 10px */
    text-transform: none;  /* Remove uppercase */
    letter-spacing: normal;  /* Remove letter spacing */
}
```

### Restore Header Height

```css
.app-header {
    padding: 12px 16px 8px;  /* Change back from 4px 12px */
    min-height: auto;  /* Remove min-height */
}

.app-header h1 {
    font-size: 18px;  /* Change back from 12px */
}

.app-header .subtitle {
    display: block;  /* Change back from none */
}
```

### Restore Form Spacing

```css
.form-row {
    gap: 12px;  /* Change back from 10px */
    margin-bottom: 12px;  /* Change back from 8px */
}

.form-group {
    gap: 4px;  /* Change back from 3px */
}
```

### Restore Console Size

```css
.console-output {
    font-size: 12px;  /* Change back from 11px */
    padding: 10px;  /* Change back from 8px */
    line-height: 1.5;  /* Change back from 1.45 */
    min-height: 250px;  /* Change back from 280px */
}
```

---

## File Locations

**Modified Files:**
- `C:\BIN\git\WhisperJav_V1_Minami_Edition\whisperjav\webview_gui\assets\style.css`
- `C:\BIN\git\WhisperJav_V1_Minami_Edition\whisperjav\webview_gui\assets\index.html`

**Backup Files:**
- `C:\BIN\git\WhisperJav_V1_Minami_Edition\whisperjav\webview_gui\assets\style.pre-aggressive.css`

**Documentation:**
- `C:\BIN\git\WhisperJav_V1_Minami_Edition\UI_AGGRESSIVE_COMPACTNESS.md`
- `C:\BIN\git\WhisperJav_V1_Minami_Edition\AGGRESSIVE_CSS_CHANGES_SUMMARY.md` (this file)

---

**Last Updated:** 2025-10-30
**Author:** Claude Code (Anthropic)
**Version:** 1.0
