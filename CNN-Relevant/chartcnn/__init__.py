"""chartcnn: chart-image CNN baseline for 5-minute US equity bars (CNN78u bundle)."""
from .data import Store, build_split, ChartDataset, SeriesDataset, make_dataset
from .models import ChartCNN, SeriesCNN1D, INPUT_KIND, build_model, parameter_table, print_parameter_count

__version__ = "2026-11-13.4"   # bump on every change; printed so a stale import is easy to spot
print(f"chartcnn {__version__} loaded from {__file__}")
