# verify_setup.py
"""
Run this before anything else to check your environment is correct.
  python verify_setup.py
"""
import sys

REQUIRED = [
    ("sqlalchemy",   "SQLAlchemy"),
    ("pandas",       "Pandas"),
    ("numpy",        "NumPy"),
    ("scipy",        "SciPy"),
    ("sklearn",      "scikit-learn"),
    ("streamlit",    "Streamlit"),
    ("plotly",       "Plotly"),
    ("networkx",     "NetworkX"),
    ("yaml",         "PyYAML"),
    ("rich",         "Rich"),
    ("typer",        "Typer"),
    ("dotenv",       "python-dotenv"),
    ("pydantic",     "Pydantic"),
    ("anthropic",    "Anthropic"),
]

print("\n🔍 Checking required packages...\n")
all_ok = True
for module, name in REQUIRED:
    try:
        __import__(module)
        print(f"  ✅  {name:<20}")
    except ImportError:
        print(f"  ❌  {name:<20}  ← MISSING: pip install {module}")
        all_ok = False

print()
if all_ok:
    print("✅ All packages installed. You're ready to run the agent!\n")
    print("Next steps:")
    print("  1. python create_demo_db.py")
    print("  2. python main.py")
    print("  3. streamlit run ui/dashboard.py\n")
else:
    print("❌ Fix missing packages then re-run this script.\n")
    sys.exit(1)