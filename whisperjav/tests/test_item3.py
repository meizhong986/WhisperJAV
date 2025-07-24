import logging
from whisperjav.modules.repetition_cleaner import RepetitionCleaner
from whisperjav.config.sanitization_constants import RepetitionConstants
from whisperjav.config.sanitization_config import SanitizationConfig

# Basic logger setup so we don't crash
logging.basicConfig(level=logging.DEBUG)

def run_isolated_test():
    """
    This test focuses ONLY on the RepetitionCleaner and the single problematic line
    to see if the corruption happens in isolation.
    """
    print("--- Running Isolated Test for Item 3 & 4 ---")

    # 1. Setup the cleaner exactly as it would be in the main script
    config = SanitizationConfig() # Use default config
    constants = config.get_effective_constants()
    repetition_constants = constants['repetition']
    cleaner = RepetitionCleaner(repetition_constants)
    
    # 2. Define the problematic line
    line_to_clean = "全雷なのか、チンチン舐められるか、どっちがいい?"
    print(f"\nOriginal Text: '{line_to_clean}'")

    # 3. Run the cleaner on ONLY this line
    cleaned_text, modifications = cleaner.clean_repetitions(line_to_clean)

    # 4. Print the result
    print(f"Cleaned Text:  '{cleaned_text}'")
    
    # 5. Check if the bug was reproduced
    expected_text = "チンチン舐められるか、チンチン舐められるか、どっちがいい?"
    if cleaned_text == expected_text:
        print("\n🔴 BUG REPRODUCED: The corruption happened even in isolation.")
    else:
        print("\n🟢 BUG NOT REPRODUCED: The corruption did not happen in isolation.")
        print("   This suggests something external to the RepetitionCleaner is the cause.")

if __name__ == "__main__":
    run_isolated_test()