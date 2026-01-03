#!/usr/bin/env python3
"""
Frontend validation tests for WhisperJAV GUI Update Check feature.

Tests the HTML structure, CSS selectors, and JavaScript syntax for the
"Check for Updates" modal and menu integration.

These tests validate:
- Required HTML elements exist with correct IDs
- CSS classes are properly defined
- JavaScript UpdateCheckManager has required methods
- Event handlers reference existing elements

Run with: pytest tests/test_update_check_frontend.py -v
"""

import os
import re
import pytest
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def assets_dir():
    """Get the path to the webview_gui assets directory."""
    repo_root = Path(__file__).parent.parent
    return repo_root / "whisperjav" / "webview_gui" / "assets"


@pytest.fixture
def index_html(assets_dir):
    """Load the index.html content."""
    html_path = assets_dir / "index.html"
    assert html_path.exists(), f"index.html not found at {html_path}"
    return html_path.read_text(encoding="utf-8")


@pytest.fixture
def app_js(assets_dir):
    """Load the app.js content."""
    js_path = assets_dir / "app.js"
    assert js_path.exists(), f"app.js not found at {js_path}"
    return js_path.read_text(encoding="utf-8")


@pytest.fixture
def style_css(assets_dir):
    """Load the style.css content."""
    css_path = assets_dir / "style.css"
    assert css_path.exists(), f"style.css not found at {css_path}"
    return css_path.read_text(encoding="utf-8")


# =============================================================================
# HTML Structure Tests
# =============================================================================

class TestHtmlStructure:
    """Tests for required HTML elements in index.html."""

    def test_theme_menu_exists(self, index_html):
        """Test that theme menu container exists."""
        assert 'id="themeMenu"' in index_html

    def test_check_updates_button_exists(self, index_html):
        """Test that Check for Updates button exists."""
        assert 'id="checkUpdatesBtn"' in index_html

    def test_check_updates_button_in_theme_menu(self, index_html):
        """Test that button is inside theme menu."""
        # Extract theme menu content
        menu_match = re.search(r'id="themeMenu"[^>]*>(.*?)</div>\s*</div>', index_html, re.DOTALL)
        assert menu_match, "Could not find themeMenu content"
        menu_content = menu_match.group(1)
        assert 'checkUpdatesBtn' in menu_content

    def test_theme_menu_divider_exists(self, index_html):
        """Test that menu divider exists before update check option."""
        assert 'class="theme-menu-divider"' in index_html

    def test_update_check_icon_exists(self, index_html):
        """Test that update icon span exists."""
        assert 'class="update-check-icon"' in index_html

    def test_modal_overlay_exists(self, index_html):
        """Test that update check modal overlay exists."""
        assert 'id="updateCheckModal"' in index_html

    def test_modal_close_button_exists(self, index_html):
        """Test that modal close button exists."""
        assert 'id="updateCheckModalClose"' in index_html

    def test_loading_state_exists(self, index_html):
        """Test that loading state container exists."""
        assert 'id="updateCheckLoading"' in index_html

    def test_result_state_exists(self, index_html):
        """Test that result state container exists."""
        assert 'id="updateCheckResult"' in index_html

    def test_error_state_exists(self, index_html):
        """Test that error state container exists."""
        assert 'id="updateCheckError"' in index_html

    def test_version_elements_exist(self, index_html):
        """Test that version display elements exist."""
        required_ids = [
            "updateCurrentVersion",
            "updateLatestVersion",
            "updateTypeBadge",
        ]
        for element_id in required_ids:
            assert f'id="{element_id}"' in index_html, f"Missing element: {element_id}"

    def test_up_to_date_message_exists(self, index_html):
        """Test that up-to-date message exists."""
        assert 'id="updateUpToDate"' in index_html

    def test_update_available_section_exists(self, index_html):
        """Test that update available section exists."""
        assert 'id="updateAvailableSection"' in index_html

    def test_release_notes_container_exists(self, index_html):
        """Test that release notes container exists."""
        assert 'id="updateReleaseNotes"' in index_html

    def test_action_buttons_exist(self, index_html):
        """Test that action buttons exist."""
        required_buttons = [
            "updateCheckLater",
            "updateCheckNow",
            "updateCheckDownload",
        ]
        for button_id in required_buttons:
            assert f'id="{button_id}"' in index_html, f"Missing button: {button_id}"

    def test_modal_has_correct_class(self, index_html):
        """Test that modal uses correct overlay class."""
        assert 'class="modal-overlay" id="updateCheckModal"' in index_html or \
               'id="updateCheckModal" class="modal-overlay"' in index_html


# =============================================================================
# CSS Validation Tests
# =============================================================================

class TestCssClasses:
    """Tests for required CSS classes in style.css."""

    def test_theme_menu_divider_style(self, style_css):
        """Test that theme-menu-divider class is defined."""
        assert ".theme-menu-divider" in style_css

    def test_update_check_option_style(self, style_css):
        """Test that update-check-option class is defined."""
        assert ".update-check-option" in style_css

    def test_update_check_icon_style(self, style_css):
        """Test that update-check-icon class is defined."""
        assert ".update-check-icon" in style_css

    def test_update_check_loading_style(self, style_css):
        """Test that update-check-loading class is defined."""
        assert ".update-check-loading" in style_css

    def test_spinner_style(self, style_css):
        """Test that spinner class is defined."""
        assert ".spinner" in style_css or ".update-check-loading .spinner" in style_css

    def test_version_info_styles(self, style_css):
        """Test that version info styles are defined."""
        assert ".version-info" in style_css
        assert ".version-row" in style_css

    def test_update_badge_styles(self, style_css):
        """Test that update badge styles are defined."""
        assert ".update-badge" in style_css
        assert ".update-badge.patch" in style_css
        assert ".update-badge.minor" in style_css
        assert ".update-badge.major" in style_css
        assert ".update-badge.critical" in style_css

    def test_up_to_date_style(self, style_css):
        """Test that up-to-date message style is defined."""
        assert ".update-up-to-date" in style_css

    def test_release_notes_styles(self, style_css):
        """Test that release notes styles are defined."""
        assert ".release-notes-details" in style_css
        assert ".release-notes-content" in style_css

    def test_update_check_error_style(self, style_css):
        """Test that error state style is defined."""
        assert ".update-check-error" in style_css

    def test_spin_animation_exists(self, style_css):
        """Test that spin keyframe animation exists."""
        assert "@keyframes spin" in style_css


class TestCssColorScheme:
    """Tests for consistent color scheme in badge styles."""

    def test_patch_badge_color(self, style_css):
        """Test that patch badge has green color."""
        # Find the patch badge rule
        match = re.search(r'\.update-badge\.patch\s*\{([^}]+)\}', style_css)
        assert match, "Patch badge style not found"
        assert "#28a745" in match.group(1) or "green" in match.group(1).lower()

    def test_minor_badge_color(self, style_css):
        """Test that minor badge has blue color."""
        match = re.search(r'\.update-badge\.minor\s*\{([^}]+)\}', style_css)
        assert match, "Minor badge style not found"
        assert "#17a2b8" in match.group(1) or "blue" in match.group(1).lower()

    def test_major_badge_color(self, style_css):
        """Test that major badge has orange color."""
        match = re.search(r'\.update-badge\.major\s*\{([^}]+)\}', style_css)
        assert match, "Major badge style not found"
        assert "#fd7e14" in match.group(1) or "orange" in match.group(1).lower()

    def test_critical_badge_color(self, style_css):
        """Test that critical badge has red color."""
        match = re.search(r'\.update-badge\.critical\s*\{([^}]+)\}', style_css)
        assert match, "Critical badge style not found"
        assert "#dc3545" in match.group(1) or "red" in match.group(1).lower()


# =============================================================================
# JavaScript Structure Tests
# =============================================================================

class TestJavaScriptStructure:
    """Tests for JavaScript UpdateCheckManager in app.js."""

    def test_update_check_manager_exists(self, app_js):
        """Test that UpdateCheckManager object is defined."""
        assert "const UpdateCheckManager" in app_js or "var UpdateCheckManager" in app_js

    def test_init_method_exists(self, app_js):
        """Test that init() method exists."""
        # Find UpdateCheckManager definition
        manager_match = re.search(
            r'const UpdateCheckManager\s*=\s*\{(.*?)\n\};',
            app_js,
            re.DOTALL
        )
        assert manager_match, "UpdateCheckManager not found"
        manager_body = manager_match.group(1)
        assert "init()" in manager_body or "init:" in manager_body

    def test_show_method_exists(self, app_js):
        """Test that show() method exists."""
        assert re.search(r'(async\s+)?show\s*\(\s*\)', app_js)

    def test_close_method_exists(self, app_js):
        """Test that close() method exists."""
        assert "close()" in app_js or "close:" in app_js

    def test_show_loading_method_exists(self, app_js):
        """Test that showLoading() method exists."""
        assert "showLoading()" in app_js or "showLoading:" in app_js

    def test_show_result_method_exists(self, app_js):
        """Test that showResult() method exists."""
        assert "showResult(" in app_js

    def test_show_error_method_exists(self, app_js):
        """Test that showError() method exists."""
        assert "showError(" in app_js

    def test_start_update_method_exists(self, app_js):
        """Test that startUpdate() method exists."""
        assert "startUpdate()" in app_js or "startUpdate:" in app_js

    def test_open_download_page_method_exists(self, app_js):
        """Test that openDownloadPage() method exists."""
        assert "openDownloadPage()" in app_js or "openDownloadPage:" in app_js

    def test_parse_markdown_method_exists(self, app_js):
        """Test that parseMarkdown() method exists."""
        assert "parseMarkdown(" in app_js

    def test_init_called_on_dom_ready(self, app_js):
        """Test that UpdateCheckManager.init() is called on DOMContentLoaded."""
        # Find the DOMContentLoaded handler
        dom_ready_match = re.search(
            r"DOMContentLoaded.*?\{(.*?)\}\s*\)",
            app_js,
            re.DOTALL
        )
        assert dom_ready_match, "DOMContentLoaded handler not found"
        handler_body = dom_ready_match.group(1)
        assert "UpdateCheckManager.init()" in handler_body


class TestJavaScriptApiCalls:
    """Tests for correct API call usage in JavaScript."""

    def test_check_for_updates_api_call(self, app_js):
        """Test that check_for_updates API is called correctly."""
        # Should call with force=true for modal
        assert "pywebview.api.check_for_updates" in app_js
        assert "check_for_updates(true)" in app_js

    def test_open_url_api_call(self, app_js):
        """Test that open_url API is called for download page."""
        assert "pywebview.api.open_url" in app_js
        # Should include GitHub releases URL
        assert "github.com/meizhong986/WhisperJAV/releases" in app_js

    def test_start_update_delegation(self, app_js):
        """Test that startUpdate delegates to UpdateManager."""
        # Should call UpdateManager.startUpdate() not api.start_update directly
        assert "UpdateManager.startUpdate()" in app_js


class TestJavaScriptEventHandlers:
    """Tests for event handler setup in JavaScript."""

    def test_menu_button_click_handler(self, app_js):
        """Test that menu button has click handler."""
        assert "checkUpdatesBtn" in app_js
        assert "addEventListener" in app_js

    def test_modal_close_button_handler(self, app_js):
        """Test that modal close button has handler."""
        assert "updateCheckModalClose" in app_js

    def test_later_button_handler(self, app_js):
        """Test that Later button has handler."""
        assert "updateCheckLater" in app_js

    def test_update_now_button_handler(self, app_js):
        """Test that Update Now button has handler."""
        assert "updateCheckNow" in app_js

    def test_download_button_handler(self, app_js):
        """Test that Download button has handler."""
        assert "updateCheckDownload" in app_js

    def test_escape_key_handler(self, app_js):
        """Test that Escape key closes modal."""
        assert "Escape" in app_js

    def test_overlay_click_closes_modal(self, app_js):
        """Test that clicking overlay closes modal."""
        # Should check if click target is the modal itself
        assert "e.target === this.modal" in app_js or \
               "event.target === this.modal" in app_js


class TestJavaScriptLogic:
    """Tests for JavaScript business logic."""

    def test_major_update_shows_download_button(self, app_js):
        """Test that major updates show Download button."""
        # Check for level === 'major' condition
        assert "'major'" in app_js
        assert "updateCheckDownload" in app_js

    def test_patch_minor_show_update_button(self, app_js):
        """Test that patch/minor updates show Update Now button."""
        assert "updateCheckNow" in app_js

    def test_notification_level_badge_update(self, app_js):
        """Test that badge class is updated based on level."""
        assert "update-badge" in app_js
        assert "className" in app_js or "classList" in app_js

    def test_version_display_formatting(self, app_js):
        """Test that versions are prefixed with 'v'."""
        assert "'v' + result.current_version" in app_js or \
               '"v" + result.current_version' in app_js

    def test_up_to_date_display_logic(self, app_js):
        """Test that up-to-date message shows when no update."""
        assert "updateUpToDate" in app_js
        # Should show when update_available is false
        assert "update_available" in app_js.lower() or "updateavailable" in app_js.lower()


# =============================================================================
# Theme Compatibility Tests
# =============================================================================

class TestThemeCompatibility:
    """Tests for theme file compatibility."""

    @pytest.fixture
    def theme_files(self, assets_dir):
        """Get all theme CSS files."""
        return list(assets_dir.glob("style.*.css"))

    def test_themes_import_base_styles(self, assets_dir):
        """Test that theme files import base style.css."""
        theme_files = [
            "style.google.css",
            "style.carbon.css",
            "style.primer.css",
        ]
        for theme_file in theme_files:
            theme_path = assets_dir / theme_file
            if theme_path.exists():
                content = theme_path.read_text(encoding="utf-8")
                assert '@import url("style.css")' in content or \
                       "@import url('style.css')" in content, \
                       f"{theme_file} should import base styles"

    def test_css_uses_variables(self, style_css):
        """Test that update check styles use CSS variables where appropriate."""
        # Key elements should use CSS variables for theme compatibility
        update_section_match = re.search(
            r'/\* Update Check Modal \*/.*?(?=/\*|$)',
            style_css,
            re.DOTALL
        )
        if update_section_match:
            update_css = update_section_match.group(0)
            # Should use variables like var(--border-color)
            assert "var(--" in update_css


# =============================================================================
# Accessibility Tests
# =============================================================================

class TestAccessibility:
    """Tests for accessibility features."""

    def test_modal_close_has_aria_label(self, index_html):
        """Test that close button has aria-label."""
        close_match = re.search(r'id="updateCheckModalClose"[^>]*>', index_html)
        assert close_match, "Close button not found"
        close_tag = close_match.group(0)
        assert 'aria-label' in close_tag

    def test_menu_items_have_role(self, index_html):
        """Test that menu items have proper role."""
        # Check updates button should have role="menuitem"
        assert 'id="checkUpdatesBtn"' in index_html
        btn_match = re.search(r'id="checkUpdatesBtn"[^>]*>', index_html)
        if btn_match:
            btn_tag = btn_match.group(0)
            assert 'role="menuitem"' in btn_tag

    def test_buttons_have_labels(self, index_html):
        """Test that buttons have visible text or aria-label."""
        # All action buttons should have text content
        assert "Update Now" in index_html
        assert "Download" in index_html
        assert "Close" in index_html or "Later" in index_html


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for consistency between HTML, CSS, and JS."""

    def test_html_ids_match_js_selectors(self, index_html, app_js):
        """Test that HTML element IDs match JavaScript selectors."""
        html_ids = [
            "updateCheckModal",
            "updateCheckModalClose",
            "checkUpdatesBtn",
            "updateCheckLoading",
            "updateCheckResult",
            "updateCheckError",
            "updateCurrentVersion",
            "updateLatestVersion",
            "updateTypeBadge",
            "updateUpToDate",
            "updateAvailableSection",
            "updateReleaseNotes",
            "updateCheckLater",
            "updateCheckNow",
            "updateCheckDownload",
        ]

        for html_id in html_ids:
            assert f'id="{html_id}"' in index_html, f"HTML missing id: {html_id}"
            # JS should reference this ID
            assert html_id in app_js, f"JS not referencing id: {html_id}"

    def test_css_classes_match_html(self, index_html, style_css):
        """Test that CSS classes used in HTML are defined."""
        html_classes = [
            "theme-menu-divider",
            "update-check-option",
            "update-check-icon",
            "update-check-loading",
            "update-check-result",
            "update-check-error",
            "version-info",
            "version-row",
            "update-badge",
            "update-up-to-date",
            "release-notes-details",
            "release-notes-content",
        ]

        for css_class in html_classes:
            # Check if class is used in HTML (may be part of multi-class declaration)
            assert css_class in index_html, f"HTML should use class: {css_class}"
            # Check if class is defined in CSS
            assert f".{css_class}" in style_css, f"CSS should define class: {css_class}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
