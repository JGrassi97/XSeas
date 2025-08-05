<p align="center">
  <img src="https://github.com/JGrassi97/XSeas/blob/main/img/logo_chatgpt.png?raw=true" width="240" height="120">
</p>

# XSeas - Xarray-based Seasonal Detection Tools

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-alpha-red.svg)]()

**XSeas** is a comprehensive Python package for detecting and analyzing meteorological seasons using machine learning and advanced clustering techniques. Built on top of xarray, it provides powerful tools for seasonal pattern analysis in climate datasets, with a focus on automating the complete workflow from data preprocessing to model training.

## 🌟 Features

### Core Algorithms
- **Radially Constrained Clustering (RCC)**: Specialized clustering algorithm for identifying seasonal breakpoints with circular time constraints
- **XRCC**: Distributed, xarray-integrated version of RCC for large spatial datasets
- **Perceptron Models**: Neural network-based seasonal classification and prediction
- **Rolling Normalization**: Advanced preprocessing techniques for climate data

### Workflow Management
- **Automated Pipeline**: Complete end-to-end workflow automation
- **Project Management**: Organized project structure with configuration management
- **Data Integration**: Support for ERA5 reanalysis and CMIP6 climate model data
- **Command Line Interface**: Easy-to-use CLI for project management and execution

### Analysis Capabilities
- **Seasonal Detection**: Identify meteorological seasons from multivariate climate data
- **Climate Projections**: Apply trained models to future climate scenarios
- **Performance Metrics**: Comprehensive evaluation tools for model validation
- **Visualization**: Built-in plotting and analysis tools

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/JGrassi97/XSeas.git
cd XSeas
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

### Creating Your First Project

```bash
# Create a new XSeas project
xseas create ./my_seasonal_analysis

# Check project status
xseas status ./my_seasonal_analysis

# Run the complete workflow
xseas run ./my_seasonal_analysis --step all
```

### Python API

```python
from xseas import SeasonDetect

# Initialize project
project = SeasonDetect("/path/to/project", "config.yaml")

# Check data availability
project.check_data_availability()

# Run complete workflow
project.run_full_workflow()

# Or run individual steps
project.prenormalize_ERA5()
project.prenormalize_CMIP6()
project.perform_clustering()
project.train_perceptron_models()
```

## 📁 Project Structure

XSeas organizes your analysis in a standardized directory structure:

```
my_project/
├── config/
│   └── config.yaml          # Project configuration
├── data/
│   ├── ERA5/                # ERA5 reanalysis data
│   │   ├── 2t/final.nc     # 2-meter temperature
│   │   ├── tp/final.nc     # Total precipitation
│   │   └── ...
│   └── CMIP6/              # CMIP6 climate model data
│       ├── ACCESS-CM2/
│       ├── EC-Earth3/
│       └── ...
├── clusterings/            # Seasonal clustering results
├── perceptron_model/       # Trained neural network models
├── projections/           # Future climate projections
├── results/              # Analysis outputs
└── notebooks/           # Jupyter notebooks
```

## ⚙️ Configuration

XSeas uses YAML configuration files to manage analysis parameters:

```yaml
name: monsoon_analysis

# Variables to analyze
variables:
  - 2m_temperature
  - total_precipitation
  - u850
  - v850

# Climate models
CMIP6_models:
  - ACCESS-CM2
  - EC-Earth3
  - CESM2-WACCM

# Variable codes
ERA_variable_code: [2t, tp, u, v]
CMIP_variable_code: [tas, pr, ua, va]

# Analysis parameters
n_seasons: 2
clustering_params:
  n_iters: 500
  starting_breakpoints: [160, 280]
  learning_rate: 1
  min_len: 10
  weights: [1, 1, 0.5, 0.5]

perceptron_params:
  n_years_training: 50
  epochs: 100
```

## 🔬 Scientific Background

### Radially Constrained Clustering (RCC)
RCC is a specialized clustering algorithm designed for seasonal time series analysis. It identifies optimal breakpoints in annual cycles while respecting:
- **Circular time constraints**: Seasons can wrap around the year boundary
- **Minimum season length**: Ensures physically meaningful seasons
- **Temporal continuity**: Maintains temporal coherence in seasonal transitions

### Applications
- **Monsoon Analysis**: Detect monsoon onset/withdrawal patterns
- **Agricultural Seasons**: Identify growing seasons and crop cycles
- **Climate Variability**: Analyze seasonal shifts under climate change
- **Extreme Events**: Study seasonal patterns in extreme weather

## 📊 Supported Data

### ERA5 Reanalysis
- Temperature (2-meter, surface)
- Precipitation (total, convective)
- Wind components (u, v at various levels)
- Pressure levels and derived variables

### CMIP6 Climate Models
- Historical simulations (1850-2014)
- Future scenarios (SSP1-2.6, SSP2-4.5, SSP5-8.5)
- Multi-model ensemble analysis
- Bias correction and preprocessing

## 📚 Documentation

### Notebooks and Examples
The `notebooks/` directory contains comprehensive examples:
- **RCC - defining seasons.ipynb**: Basic seasonal clustering
- **Perceptron - tracking seasons.ipynb**: Neural network training
- **Manager - automating the pipeline.ipynb**: Complete workflow automation

### API Reference
Detailed documentation for all classes and functions:
- **SeasonDetect**: Main workflow manager
- **RCC/XRCC**: Clustering algorithms
- **Normalization**: Preprocessing utilities
- **Perceptron**: Neural network models

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/JGrassi97/XSeas.git
cd XSeas

# Create conda environment
conda env create -f environment.yml
conda activate xseas

# Install in development mode
pip install -e .[dev]
```

### Running Tests

```bash
# Run test suite
pytest tests/

# Run with coverage
pytest --cov=xseas tests/
```

### Code Style

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code standards and style
- Testing requirements
- Documentation guidelines
- Pull request process

### Areas for Contribution
- **New Algorithms**: Additional clustering or classification methods
- **Data Formats**: Support for new climate data sources
- **Visualizations**: Enhanced plotting and analysis tools
- **Performance**: Optimization for large datasets
- **Documentation**: Examples, tutorials, and API documentation

## 📈 Performance

XSeas is designed for scalability:
- **Dask Integration**: Parallel processing for large datasets
- **Memory Efficient**: Chunked processing and lazy evaluation
- **GPU Support**: Compatible with GPU-accelerated computing
- **Cloud Ready**: Works with cloud-based data storage

## 🏆 Citation

If you use XSeas in your research, please cite:

```bibtex
@software{grassi2024xseas,
  author = {Grassi, Jacopo},
  title = {XSeas: Xarray-based tools for meteorological Seasons detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/JGrassi97/XSeas}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **xarray**: For providing the foundation for labeled data manipulation
- **scikit-learn**: For machine learning utilities and metrics
- **dask**: For parallel and distributed computing capabilities
- **Climate Data Community**: For open data and collaborative research

## 📞 Support and Contact

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/JGrassi97/XSeas/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/JGrassi97/XSeas/discussions)
- **Email**: For direct contact, reach out to [jacopo.grassi@unibo.it](mailto:jacopo.grassi@unibo.it)

---

**XSeas** - Advancing seasonal analysis in climate science through machine learning and automation. 🌍✨
