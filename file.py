"""
Original babel_string function - now integrated with the full Babel Library project.

For the full implementation, see:
- dataset.py: generate_babel_dataset() for generating datasets
- babel_library.py: BabelEBM for the Energy-Based Model
- babel_exploration.ipynb: Interactive Jupyter notebook
"""

import random
import string

def babel_string(length=40):
    """
    Generate a random Babel string (original simple version).
    
    For more advanced usage, see dataset.py:
    - generate_babel_strings() for multiple sequences
    - generate_babel_dataset() for index arrays
    """
    return ''.join(random.choice(string.ascii_lowercase + " ") for _ in range(length))
