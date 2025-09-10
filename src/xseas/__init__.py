from .models import RCC, train_perceptron, train_perceptron_spatial, predict_custom, evaluate_custom
from .xarray import XRCC, X_labels, tile_labels
from .manager import (
    SeasonDetect, 
    load_variables, 
    create_directory_structure,
    get_sample_config,
    create_sample_config
)

# Version information
__version__ = "0.0.1"
__author__ = "Jacopo Grassi"
__email__ = "jacopo.grassi@example.com"  # Update with actual email

# Package description
__description__ = (
    "Xarray-based tools for meteorological Seasons detection (XSeas) - "
    "AI & ML based algorithms for seasonal pattern analysis in climate data"
)