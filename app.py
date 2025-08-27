import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils.calculator import Calculator
from utils.graphing import GraphingTool
from utils.statistics import StatisticalAnalysis
from utils.equation_solver import EquationSolver
from utils.constants import MathConstants
from utils.unit_converter import UnitConverter

# Set page configuration
st.set_page_config(
    page_title="Mathematical Toolkit",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize utility classes
calculator = Calculator()
graphing_tool = GraphingTool()
stats_analyzer = StatisticalAnalysis()
equation_solver = EquationSolver()
math_constants = MathConstants()
unit_converter = UnitConverter()

# Main title
st.title("🧮 Mathematical Toolkit")
st.markdown("A comprehensive mathematical toolkit with calculation, graphing, and analysis capabilities")

# Sidebar navigation
st.sidebar.title("Navigation")
selected_tool = st.sidebar.selectbox(
    "Select a Mathematical Tool:",
    [
        "Basic Calculator",
        "Function Graphing",
        "Statistical Analysis",
        "Equation Solver",
        "Mathematical Constants",
        "Unit Converter"
    ]
)

# Basic Calculator
if selected_tool == "Basic Calculator":
    st.header("🔢 Basic Calculator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        expression = st.text_input(
            "Enter mathematical expression:",
            placeholder="e.g., 2 + 3 * 4, sqrt(16), sin(pi/2), log(10)"
        )
        
        if st.button("Calculate", type="primary"):
            if expression:
                result = calculator.evaluate_expression(expression)
                if result is not None:
                    st.success(f"Result: {result}")
                else:
                    st.error("Invalid expression. Please check your input.")
    
    with col2:
        st.subheader("Quick Operations")
        
        # Quick arithmetic operations
        num1 = st.number_input("Number 1:", value=0.0)
        operation = st.selectbox("Operation:", ["+", "-", "*", "/", "^"])
        num2 = st.number_input("Number 2:", value=0.0)
        
        if st.button("Calculate Quick"):
            result = calculator.quick_operation(num1, operation, num2)
            if result is not None:
                st.write(f"{num1} {operation} {num2} = {result}")
            else:
                st.error("Invalid operation")
    
    # Calculator history
    if hasattr(calculator, 'history') and calculator.history:
        st.subheader("Recent Calculations")
        for calc in calculator.history[-5:]:
            st.write(f"• {calc}")

# Function Graphing
elif selected_tool == "Function Graphing":
    st.header("📈 Function Graphing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Function Settings")
        
        function_expr = st.text_input(
            "Enter function f(x):",
            value="x**2",
            placeholder="e.g., x**2, sin(x), exp(x), log(x)"
        )
        
        x_min = st.number_input("X minimum:", value=-10.0)
        x_max = st.number_input("X maximum:", value=10.0)
        
        graph_type = st.selectbox("Graph Type:", ["Line Plot", "Scatter Plot"])
        
        if st.button("Generate Graph", type="primary"):
            fig = graphing_tool.plot_function(function_expr, x_min, x_max, graph_type.lower())
            if fig:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Invalid function expression")
    
    with col2:
        if not st.session_state.get('graph_generated', False):
            st.info("Enter a function and click 'Generate Graph' to see the visualization")
    
    # Multiple functions comparison
    st.subheader("Compare Multiple Functions")
    functions = []
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(3):
        func = st.text_input(f"Function {i+1}:", key=f"func_{i}", placeholder=f"e.g., x**{i+1}")
        if func:
            functions.append((func, colors[i]))
    
    if functions and st.button("Compare Functions"):
        fig = graphing_tool.plot_multiple_functions(functions, x_min, x_max)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Statistical Analysis
elif selected_tool == "Statistical Analysis":
    st.header("📊 Statistical Analysis")
    
    # Data input options
    data_input_method = st.radio("Choose data input method:", ["Manual Entry", "Upload CSV"])
    
    data = None
    
    if data_input_method == "Manual Entry":
        data_input = st.text_area(
            "Enter data (comma-separated values):",
            placeholder="1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
        )
        
        if data_input:
            try:
                data = [float(x.strip()) for x in data_input.split(',') if x.strip()]
            except ValueError:
                st.error("Please enter valid numbers separated by commas")
    
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                selected_column = st.selectbox("Select column for analysis:", numeric_columns)
                data = df[selected_column].dropna().tolist()
    
    if data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Descriptive Statistics")
            stats = stats_analyzer.calculate_descriptive_stats(data)
            
            for stat_name, value in stats.items():
                st.metric(stat_name.title(), f"{value:.4f}")
        
        with col2:
            st.subheader("Data Distribution")
            fig = stats_analyzer.create_histogram(data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analyses
        st.subheader("Additional Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Box Plot"):
                fig = stats_analyzer.create_boxplot(data)
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            if st.button("Normal Distribution Test"):
                result = stats_analyzer.normality_test(data)
                st.write(f"Shapiro-Wilk test p-value: {result:.6f}")
                if result > 0.05:
                    st.success("Data appears to be normally distributed (p > 0.05)")
                else:
                    st.warning("Data may not be normally distributed (p ≤ 0.05)")

# Equation Solver
elif selected_tool == "Equation Solver":
    st.header("⚖️ Equation Solver")
    
    equation_type = st.selectbox("Select equation type:", ["Linear", "Quadratic", "System of Linear Equations"])
    
    if equation_type == "Linear":
        st.subheader("Linear Equation: ax + b = 0")
        
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Coefficient a:", value=1.0)
            b = st.number_input("Coefficient b:", value=0.0)
        
        with col2:
            if st.button("Solve Linear Equation"):
                solution = equation_solver.solve_linear(a, b)
                if solution is not None:
                    st.success(f"Solution: x = {solution}")
                    
                    # Show step-by-step solution
                    st.subheader("Step-by-step solution:")
                    st.write(f"1. Start with: {a}x + {b} = 0")
                    st.write(f"2. Subtract {b} from both sides: {a}x = {-b}")
                    st.write(f"3. Divide by {a}: x = {-b}/{a} = {solution}")
                else:
                    st.error("No solution exists (coefficient 'a' cannot be zero)")
    
    elif equation_type == "Quadratic":
        st.subheader("Quadratic Equation: ax² + bx + c = 0")
        
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Coefficient a:", value=1.0, key="quad_a")
            b = st.number_input("Coefficient b:", value=0.0, key="quad_b")
            c = st.number_input("Coefficient c:", value=0.0, key="quad_c")
        
        with col2:
            if st.button("Solve Quadratic Equation"):
                solutions = equation_solver.solve_quadratic(a, b, c)
                if solutions:
                    if len(solutions) == 1:
                        st.success(f"Solution: x = {solutions[0]}")
                    else:
                        st.success(f"Solutions: x₁ = {solutions[0]}, x₂ = {solutions[1]}")
                    
                    # Show discriminant and nature of roots
                    discriminant = b**2 - 4*a*c
                    st.write(f"Discriminant (Δ) = {discriminant}")
                    if discriminant > 0:
                        st.info("Two distinct real roots")
                    elif discriminant == 0:
                        st.info("One repeated real root")
                    else:
                        st.info("Two complex conjugate roots")
                else:
                    st.error("Invalid equation (coefficient 'a' cannot be zero for quadratic)")
    
    else:  # System of Linear Equations
        st.subheader("System of Linear Equations")
        st.write("Solve system: a₁x + b₁y = c₁ and a₂x + b₂y = c₂")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("First equation: a₁x + b₁y = c₁")
            a1 = st.number_input("a₁:", value=1.0)
            b1 = st.number_input("b₁:", value=1.0)
            c1 = st.number_input("c₁:", value=1.0)
        
        with col2:
            st.write("Second equation: a₂x + b₂y = c₂")
            a2 = st.number_input("a₂:", value=1.0)
            b2 = st.number_input("b₂:", value=-1.0)
            c2 = st.number_input("c₂:", value=1.0)
        
        if st.button("Solve System"):
            solution = equation_solver.solve_system([[a1, b1], [a2, b2]], [c1, c2])
            if solution:
                st.success(f"Solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}")
            else:
                st.error("No unique solution exists (system may be inconsistent or dependent)")

# Mathematical Constants
elif selected_tool == "Mathematical Constants":
    st.header("🔢 Mathematical Constants")
    
    constants = math_constants.get_all_constants()
    
    # Display constants in a grid
    cols = st.columns(3)
    
    for i, (name, info) in enumerate(constants.items()):
        with cols[i % 3]:
            st.metric(
                label=info['symbol'],
                value=f"{info['value']:.10f}",
                help=info['description']
            )
            st.caption(name)
    
    # Constant calculator
    st.subheader("Calculate with Constants")
    
    col1, col2 = st.columns(2)
    
    with col1:
        expression_with_constants = st.text_input(
            "Enter expression using constants:",
            placeholder="e.g., pi * r^2, e^(i*pi), sqrt(2) * phi"
        )
    
    with col2:
        if st.button("Calculate with Constants"):
            if expression_with_constants:
                result = math_constants.evaluate_with_constants(expression_with_constants)
                if result is not None:
                    st.success(f"Result: {result}")
                else:
                    st.error("Invalid expression")
    
    # Constants information
    st.subheader("Detailed Information")
    selected_constant = st.selectbox("Select a constant for more information:", list(constants.keys()))
    
    if selected_constant:
        info = constants[selected_constant]
        st.write(f"**{selected_constant}**")
        st.write(f"Symbol: {info['symbol']}")
        st.write(f"Value: {info['value']}")
        st.write(f"Description: {info['description']}")

# Unit Converter
elif selected_tool == "Unit Converter":
    st.header("🔄 Unit Converter")
    
    conversion_category = st.selectbox(
        "Select conversion category:",
        ["Length", "Weight", "Temperature", "Area", "Volume", "Time"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value = st.number_input("Value to convert:", value=1.0)
    
    with col2:
        from_unit = st.selectbox("From unit:", unit_converter.get_units(conversion_category))
    
    with col3:
        to_unit = st.selectbox("To unit:", unit_converter.get_units(conversion_category))
    
    if st.button("Convert", type="primary"):
        result = unit_converter.convert(value, from_unit, to_unit, conversion_category)
        if result is not None:
            st.success(f"{value} {from_unit} = {result:.6f} {to_unit}")
        else:
            st.error("Conversion failed. Please check your inputs.")
    
    # Conversion table
    st.subheader(f"{conversion_category} Conversion Reference")
    
    if st.checkbox("Show conversion table"):
        base_value = 1.0
        units = unit_converter.get_units(conversion_category)
        base_unit = units[0]  # Use first unit as base
        
        conversion_data = []
        for unit in units:
            converted = unit_converter.convert(base_value, base_unit, unit, conversion_category)
            if converted is not None:
                conversion_data.append({
                    "Unit": unit,
                    f"1 {base_unit} =": f"{converted:.6f} {unit}"
                })
        
        if conversion_data:
            import pandas as pd
            df = pd.DataFrame(conversion_data)
            st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Mathematical Toolkit v1.0")
