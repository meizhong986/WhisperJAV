"""
Test script to explore ttk.Notebook tab spacing options.
Tests different approaches to create horizontal gaps between tabs.
"""
import tkinter as tk
from tkinter import ttk

def test_default_theme():
    """Test 1: Default Windows theme with various configurations"""
    root = tk.Tk()
    root.title("Test 1: Default Theme")
    root.geometry("600x400")

    style = ttk.Style(root)
    print(f"Available themes: {style.theme_names()}")
    print(f"Current theme: {style.theme_use()}")

    # Try to create gap with borderwidth
    style.configure(
        'TNotebook.Tab',
        padding=[24, 8, 24, 8],
        borderwidth=4,              # Does this create gaps?
        bordercolor='#FFFFFF'       # White to match background
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=20, pady=20)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    notebook.add(tab1, text="Tab 1")
    notebook.add(tab2, text="Tab 2")

    ttk.Label(root, text="Default theme - borderwidth=4").pack()

    root.mainloop()

def test_clam_theme():
    """Test 2: Clam theme with borderwidth"""
    root = tk.Tk()
    root.title("Test 2: Clam Theme")
    root.geometry("600x400")

    style = ttk.Style(root)
    style.theme_use('clam')
    print(f"Using theme: {style.theme_use()}")

    # Configure notebook background
    style.configure('TNotebook', background='#FFFFFF')

    # Try borderwidth with clam theme
    style.configure(
        'TNotebook.Tab',
        padding=[24, 8, 24, 8],
        borderwidth=4,
        bordercolor='#FFFFFF',
        background='#F7F8FA'
    )

    style.map(
        'TNotebook.Tab',
        background=[('selected', '#F0F2F5'), ('!selected', '#F7F8FA')]
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=20, pady=20)

    tab1 = ttk.Frame(notebook, style='Tab.TFrame')
    tab2 = ttk.Frame(notebook, style='Tab.TFrame')
    notebook.add(tab1, text="Tab 1")
    notebook.add(tab2, text="Tab 2")

    ttk.Label(root, text="Clam theme - borderwidth=4, bordercolor white").pack()

    root.mainloop()

def test_relief_approach():
    """Test 3: Using relief parameter"""
    root = tk.Tk()
    root.title("Test 3: Relief Approach")
    root.geometry("600x400")

    style = ttk.Style(root)
    style.theme_use('clam')

    style.configure('TNotebook', background='#FFFFFF')

    style.configure(
        'TNotebook.Tab',
        padding=[24, 8, 24, 8],
        borderwidth=3,
        relief='solid',
        background='#F7F8FA'
    )

    style.map(
        'TNotebook.Tab',
        background=[('selected', '#F0F2F5'), ('!selected', '#F7F8FA')],
        relief=[('selected', 'solid'), ('!selected', 'solid')]
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=20, pady=20)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    notebook.add(tab1, text="Tab 1")
    notebook.add(tab2, text="Tab 2")

    ttk.Label(root, text="Clam theme - relief='solid', borderwidth=3").pack()

    root.mainloop()

def test_tabmargins():
    """Test 4: Using tabmargins parameter"""
    root = tk.Tk()
    root.title("Test 4: Tab Margins")
    root.geometry("600x400")

    style = ttk.Style(root)
    style.theme_use('clam')

    # tabmargins: [left, top, right, bottom]
    style.configure(
        'TNotebook',
        background='#FFFFFF',
        tabmargins=[12, 4, 12, 0]
    )

    style.configure(
        'TNotebook.Tab',
        padding=[24, 8, 24, 8],
        background='#F7F8FA'
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=20, pady=20)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    notebook.add(tab1, text="Tab 1")
    notebook.add(tab2, text="Tab 2")

    ttk.Label(root, text="Clam theme - tabmargins=[12, 4, 12, 0]").pack()

    root.mainloop()

def test_expanded_padding():
    """Test 5: Increased horizontal padding to create visual separation"""
    root = tk.Tk()
    root.title("Test 5: Expanded Padding")
    root.geometry("600x400")

    style = ttk.Style(root)
    style.theme_use('clam')

    style.configure('TNotebook', background='#FFFFFF')

    # Much larger horizontal padding
    style.configure(
        'TNotebook.Tab',
        padding=[40, 8, 40, 8],  # 40px horizontal padding
        background='#F7F8FA'
    )

    style.map(
        'TNotebook.Tab',
        background=[('selected', '#F0F2F5'), ('!selected', '#F7F8FA')]
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=20, pady=20)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    notebook.add(tab1, text="Tab 1")
    notebook.add(tab2, text="Tab 2")

    ttk.Label(root, text="Clam theme - Large horizontal padding [40, 8, 40, 8]").pack()

    root.mainloop()

if __name__ == "__main__":
    print("Running tab spacing tests...")
    print("Close each window to proceed to next test\n")

    print("\n=== Test 1: Default Theme ===")
    test_default_theme()

    print("\n=== Test 2: Clam Theme ===")
    test_clam_theme()

    print("\n=== Test 3: Relief Approach ===")
    test_relief_approach()

    print("\n=== Test 4: Tab Margins ===")
    test_tabmargins()

    print("\n=== Test 5: Expanded Padding ===")
    test_expanded_padding()

    print("\nAll tests complete!")
