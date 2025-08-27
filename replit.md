# Mathematical Toolkit

## Overview

This is a comprehensive mathematical toolkit built with Streamlit that provides various mathematical calculation and analysis capabilities. The application serves as an all-in-one solution for basic calculations, function graphing, statistical analysis, equation solving, mathematical constants reference, and unit conversions. It's designed to be user-friendly with an interactive web interface that makes mathematical operations accessible to users of all levels.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with a sidebar navigation system
- **Layout**: Wide layout configuration with expandable sidebar for tool selection
- **Page Structure**: Single-page application with conditional rendering based on selected tool
- **Visualization**: Integrated Plotly for interactive graphing and Matplotlib for additional plotting capabilities

### Backend Architecture
- **Modular Design**: Utility-based architecture with separate modules for each mathematical function
- **Class-Based Components**: Each utility is implemented as a dedicated class with specific responsibilities
- **Calculation Engine**: Uses NumPy for numerical computations and SymPy for symbolic mathematics
- **Safety Features**: Secure expression evaluation with predefined safe dictionaries for mathematical operations

### Core Mathematical Modules
- **Calculator**: Basic and advanced mathematical operations with expression evaluation
- **Graphing Tool**: Function plotting with support for various graph types and customization
- **Statistical Analysis**: Descriptive statistics, hypothesis testing, and data visualization
- **Equation Solver**: Linear, quadratic, cubic, and general equation solving capabilities
- **Constants Library**: Comprehensive collection of mathematical and physical constants
- **Unit Converter**: Multi-category unit conversion system with temperature handling

### Data Processing
- **NumPy Integration**: Efficient numerical array operations and mathematical functions
- **Pandas Support**: Data manipulation and analysis capabilities for statistical operations
- **SymPy Integration**: Symbolic mathematics for equation parsing and solving
- **SciPy Statistics**: Advanced statistical functions and probability distributions

### Security and Safety
- **Expression Evaluation**: Controlled evaluation environment with whitelisted functions and constants
- **Input Validation**: Type checking and error handling for mathematical operations
- **Safe Dictionary**: Predefined mathematical functions to prevent code injection

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Static plotting and visualization
- **Plotly**: Interactive graphing and data visualization
- **SymPy**: Symbolic mathematics and equation manipulation
- **SciPy**: Scientific computing and statistical analysis
- **Pandas**: Data manipulation and analysis

### Mathematical Dependencies
- **Math Module**: Standard Python mathematical functions
- **Statistics Module**: Built-in statistical functions
- **Typing Module**: Type hints and annotations for better code structure

### Visualization Stack
- **Plotly Graph Objects**: Advanced interactive plotting capabilities
- **Plotly Express**: Simplified plotting interface
- **Matplotlib.pyplot**: Traditional plotting functions for basic graphs