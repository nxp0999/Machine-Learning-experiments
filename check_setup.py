"""
Setup Verification Script
Run this to check if everything is installed correctly
"""

import sys

print("=" * 60)
print("CHECKING YOUR SETUP")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info >= (3, 7):
    print("   ✓ Python version is compatible")
else:
    print("   ✗ Python 3.7+ required!")

# Check required libraries
required_libraries = {
    'sklearn': 'scikit-learn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib'
}

print("\n2. Checking Required Libraries:")
missing_libraries = []

for module_name, install_name in required_libraries.items():
    try:
        if module_name == 'sklearn':
            import sklearn
            print(f"   ✓ {install_name} (version {sklearn.__version__})")
        elif module_name == 'pandas':
            import pandas
            print(f"   ✓ {install_name} (version {pandas.__version__})")
        elif module_name == 'numpy':
            import numpy
            print(f"   ✓ {install_name} (version {numpy.__version__})")
        elif module_name == 'matplotlib':
            import matplotlib
            print(f"   ✓ {install_name} (version {matplotlib.__version__})")
    except ImportError:
        print(f"   ✗ {install_name} is NOT installed")
        missing_libraries.append(install_name)

# Summary
print("\n" + "=" * 60)
if missing_libraries:
    print("SETUP INCOMPLETE!")
    print(f"\nPlease install missing libraries:")
    print(f"pip install {' '.join(missing_libraries)}")
else:
    print("✓ ALL REQUIREMENTS MET!")    
    print("\nYou're ready to start the project!")
print("=" * 60)
