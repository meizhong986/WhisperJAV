"""
Simple test to understand what creates horizontal gaps between tabs.
This test creates side-by-side comparisons to see what actually works.
"""
import tkinter as tk
from tkinter import ttk

def create_test_window(title, config_func):
    """Create a test window with given configuration"""
    root = tk.Tk()
    root.title(title)
    root.geometry("700x300")
    root.configure(bg='#FFFFFF')

    style = ttk.Style(root)

    # Apply the configuration function
    config_func(style)

    # Create notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=20, pady=20)

    # Add tabs
    for i in range(1, 4):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=f"Tab {i}")
        ttk.Label(tab, text=f"Content of Tab {i}").pack(padx=20, pady=20)

    # Add description label
    ttk.Label(root, text=title, font=('Arial', 10, 'bold')).pack(pady=5)

    return root

def test1_baseline(style):
    """Baseline: Default with no customization"""
    pass

def test2_borderwidth_only(style):
    """Test borderwidth parameter"""
    style.configure(
        'TNotebook.Tab',
        borderwidth=5,
        bordercolor='#FFFFFF'
    )

def test3_clam_with_borderwidth(style):
    """Test clam theme with borderwidth"""
    style.theme_use('clam')
    style.configure('TNotebook', background='#FFFFFF')
    style.configure(
        'TNotebook.Tab',
        borderwidth=5,
        bordercolor='#FFFFFF',
        background='#E8E8E8'
    )

def test4_extra_padding(style):
    """Test extra horizontal padding"""
    style.theme_use('clam')
    style.configure('TNotebook', background='#FFFFFF')
    style.configure(
        'TNotebook.Tab',
        padding=[30, 8],  # Extra horizontal padding
        background='#E8E8E8'
    )

def test5_tabmargins(style):
    """Test tabmargins parameter"""
    style.theme_use('clam')
    style.configure(
        'TNotebook',
        background='#FFFFFF',
        tabmargins=[10, 5, 10, 0]  # Margins around tab area
    )
    style.configure(
        'TNotebook.Tab',
        padding=[20, 8],
        background='#E8E8E8'
    )

def test6_expand_option(style):
    """Test expand option in style.map"""
    style.theme_use('clam')
    style.configure('TNotebook', background='#FFFFFF')
    style.configure(
        'TNotebook.Tab',
        padding=[20, 8],
        background='#E8E8E8'
    )
    style.map(
        'TNotebook.Tab',
        expand=[('selected', [1, 1, 2, 0])]  # Expand selected tab
    )

def main():
    tests = [
        ("1. Baseline (no customization)", test1_baseline),
        ("2. Borderwidth=5", test2_borderwidth_only),
        ("3. Clam + Borderwidth", test3_clam_with_borderwidth),
        ("4. Clam + Extra Padding", test4_extra_padding),
        ("5. Clam + Tabmargins", test5_tabmargins),
        ("6. Clam + Expand Option", test6_expand_option),
    ]

    windows = []
    for title, config_func in tests:
        try:
            window = create_test_window(title, config_func)
            windows.append(window)
        except Exception as e:
            print(f"Error in {title}: {e}")

    if windows:
        print(f"Created {len(windows)} test windows")
        print("Check each window to see the effect on tab spacing")
        print("Close all windows to finish")

        # Start main loop on first window
        windows[0].mainloop()

if __name__ == "__main__":
    print("Testing different approaches to create horizontal gaps between tabs...")
    print("=" * 70)
    main()
