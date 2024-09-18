import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_squared_error
from data_processing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to calculate color based on performance indicator
def get_performance_color(performance):
    if performance >= 90:
        return "green"
    elif performance >= 75:
        return "yellowgreen"
    elif performance >= 50:
        return "orange"
    else:
        return "red"

# Load the dataset
df = load_data()

# Preprocess the data
df = preprocess_data(df)

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Student Performance Predictor With Regression Analysis</h1>", unsafe_allow_html=True)

# Add a divider
st.divider()

# Sidebar for user input
st.sidebar.header('Select Test Size')
test_size = st.sidebar.slider('Test Size', 0.1, 0.9, 0.2)

# Splitting the data into features and target
X = df.drop(columns=['Performance Index'])
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Create the regression equation
target_variable = 'Performance Index'
equation = f"{target_variable} = {intercept:.2f}"
for i, coef in enumerate(coefficients):
    feature_name = X.columns[i]
    equation += f" + {coef:.2f} * {feature_name}"

# User inputs for predicting fridge price
st.sidebar.subheader('Select parameters for predicting student performance')

# Hours Studied
hours_studied = st.sidebar.number_input('Hours Studied', min_value=1, max_value=15, value=2)

# Previous Scores
previous_scores = st.sidebar.number_input('Previous Scores', min_value=10, max_value=100, value=68)

# Sleep hours
sleep_hours = st.sidebar.number_input('Sleep Hours', min_value=1, max_value=12, value=5)

# Practice Papers
practice_papers = st.sidebar.number_input('Practice Papers Solved', min_value=0, max_value=10, value=2)

# Extra curricular activities
activity = st.sidebar.selectbox('Extra Curricular Activies', options=['Yes', 'No'])
activity = 1 if activity == 'Yes' else 0

# Prepare input features
input_features = [[hours_studied, previous_scores, sleep_hours, practice_papers, activity]]

# Columns
col1, col2, col3 = st.columns([1.3, 1.5, 1])

# Write regression equation
with col1:
    st.markdown("<h3 style='text-align: center;'>Regression Equation</h1>",
                unsafe_allow_html=True)
    st.write(equation)

# Predict the performance indicator using the regression model
with col2:
    performance_indicator = model.predict(input_features)[0]
    formatted_performance = f"{performance_indicator:.0f}"  # No decimal places and no "k"

    # Get performance color
    performance_color = get_performance_color(performance_indicator)

    # Display predicted price with custom styling: bold, larger, and colored
    st.markdown("<h3 style='text-align: center;'>Predicted Performance Indicator</h1>",
                unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='color: {performance_color}; font-weight: bold; text-align: center;'>{formatted_performance}</h2>",
        unsafe_allow_html=True)

# Calculating Mean Squared Error
with col3:
    st.markdown("<h3 style='text-align: center;'>Mean Squared Error</h1>",
                unsafe_allow_html=True)
    mse = mean_squared_error(y_test, y_pred)
    st.markdown(f"<h4 style='text-align: center;'>{mse:.2f}</h4>", unsafe_allow_html=True)

# Add a divider
st.divider()

# Columns for plots
col1, col2 = st.columns(2)  # Create two columns

# Plot 3: Prediction vs Actual Values Plot
with col1:
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue', label='Predicted Values')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction vs Actual Values')
    ax.legend()
    st.pyplot(fig)

# Plot 4: Heatmap of Correlation Matrix
with col2:
    fig, ax = plt.subplots()
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap of Correlation Matrix')
    st.pyplot(fig)

# Add a divider
st.divider()

# Correlation between user selected variables
st.sidebar.header('Select Variables for Correlation Plot')
columns = df.columns.tolist()
variable_1 = st.sidebar.selectbox('Select the first variable', columns)
variable_2 = st.sidebar.selectbox('Select the second variable', columns)
if variable_1 != variable_2:
    # Plotting the scatter plot with regression line
    st.markdown(f"<h2 style='text-align: center;'>Scatter Plot of {variable_1} VS {variable_2}</h1>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(20, 17))  # Adjust the width and height as needed
    sns.regplot(x=df[variable_1], y=df[variable_2], ax=ax, scatter_kws={'s':50}, line_kws={'color':'red'})
    ax.set_xlabel(variable_1)
    ax.set_ylabel(variable_2)
    st.pyplot(fig)
else:
    st.write("Please select different variables for the scatter plot.")