# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "datasets" / "dataset_cell_growth_4factor.csv"
FIGURES_DIR = ROOT_DIR / "figures"
TABLES_DIR = ROOT_DIR / "tables"

# --- Column names ------------------------------------------------------
TARGET = "Specific Growth Rate [d^-1]"
FEATURES = ["Temperature [C]", "pH", "Glucose [mM]", "Lactate [mM]"]

# --- Reproducibility and validation -------------------------------------
RANDOM_STATE = 0
TEST_SIZE = 0.2
CV_FOLDS = 4

# --- Hyperparameter search spaces ----------------------------------------
POLY_DEGREES = list(range(1, 11))
KNN_NEIGHBORS = list(range(1, 41))
KNN_WEIGHTS = ["uniform", "distance"]

# --- Learning curve ------------------------------------------------------
LC_SAMPLE_SIZES = list(range(60, 301, 40)) + [318]
LC_SEED = 1

# --- Interpretation settings ------------------------------------------------
N_GRID_POINTS = 100
EXTREME_GRID_POINTS = 15  # points per feature in the 4-D search grid
EXTREME_FRACTION = 0.02  # share of reliable grid points called "high" / "low"
RELIABILITY_PERCENTILE = 95  # distance cut-off, percentile of the training data

# --- Plotting ----------------------------------------------------------------
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54
FIG_WIDTH_WIDE = 29.21 / 2.54
FONTSIZE_XY_LABEL = 14
FONTSIZE_TICKS = 12
FONTSIZE_LEGEND = 12
RELIABILITY_K = 5  # a condition is reliable if its 5th nearest sample is close