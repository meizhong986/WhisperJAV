"""Quick validation test for GUI refactoring - Step 1"""
import sys
import os
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform.startswith("win"):
    os.environ["PYTHONIOENCODING"] = "utf-8"
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from whisperjav.gui.whisperjav_gui import WhisperJAVGUI

def test_gui_instantiation():
    """Test that GUI can be instantiated without errors"""
    try:
        # Create GUI instance (don't run mainloop)
        app = WhisperJAVGUI()
        print("[PASS] GUI instantiation successful")

        # Verify notebook exists
        assert hasattr(app, 'notebook'), "Notebook not found"
        print("[PASS] Notebook widget exists")

        # Verify tabs exist
        tab_count = app.notebook.index("end")
        assert tab_count == 2, f"Expected 2 tabs, found {tab_count}"
        print(f"[PASS] Found {tab_count} tabs")

        # Verify tab names
        tab1_text = app.notebook.tab(0, "text")
        tab2_text = app.notebook.tab(1, "text")
        print(f"   Tab 1: {tab1_text}")
        print(f"   Tab 2: {tab2_text}")

        # Verify all critical variables exist
        critical_vars = [
            'mode_var', 'sens_var', 'lang_var',
            'opt_adapt_cls', 'opt_adapt_enh', 'opt_smart_post',
            'verbosity_var', 'model_override_enabled', 'model_selection_var',
            'async_var', 'workers_var', 'credit_var',
            'keep_temp_var', 'temp_var'
        ]

        for var_name in critical_vars:
            assert hasattr(app, var_name), f"Variable {var_name} not found"
        print(f"[PASS] All {len(critical_vars)} critical variables preserved")

        # Verify methods exist
        critical_methods = ['build_args', '_toggle_model_override', '_browse_temp_dir']
        for method_name in critical_methods:
            assert hasattr(app, method_name), f"Method {method_name} not found"
        print(f"[PASS] All {len(critical_methods)} critical methods preserved")

        # Test build_args with mock input
        app.inputs_listbox.insert(0, "test_video.mp4")
        try:
            args = app.build_args()
            print(f"[PASS] build_args() executes successfully")
            print(f"   Generated args: {' '.join(args[:5])}...")

            # Verify expected arguments
            assert '--mode' in args, "Missing --mode argument"
            assert '--sensitivity' in args, "Missing --sensitivity argument"
            assert '--output-dir' in args, "Missing --output-dir argument"
            print("[PASS] Expected CLI arguments present")

        except Exception as e:
            print(f"[FAIL] build_args() failed: {e}")
            return False

        # Verify toggle_advanced method removed
        assert not hasattr(app, 'toggle_advanced'), "toggle_advanced() should be removed"
        print("[PASS] Obsolete toggle_advanced() method removed")

        print("\n" + "="*50)
        print("SUCCESS: ALL VALIDATION TESTS PASSED")
        print("="*50)

        # Destroy app
        app.destroy()
        return True

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_instantiation()
    sys.exit(0 if success else 1)
