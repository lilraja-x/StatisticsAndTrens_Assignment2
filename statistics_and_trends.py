"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file, or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates a scatter plot to show the relationship between 
    customer location (Latitude & Longitude) and churn score.

    :param df: Pandas DataFrame containing the dataset.
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['Longitude'], df['Latitude'],
        c=df['Churn Score'], cmap='coolwarm', alpha=0.5
    )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Customer Location (Latitude & Longitude) vs Churn Score')
    plt.colorbar(scatter, label="Churn Score")
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    """
    Creates a bar plot to visualize the count of customers 
    based on their contract type.

    :param df: Pandas DataFrame containing the dataset.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['Contract'])
    plt.xlabel('Contract Type')
    plt.ylabel('Number of Customers')
    plt.title('Customer Count by Contract Type')
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    """
    Creates a correlation heatmap to show relationships between numerical features.

    :param df: Pandas DataFrame containing the dataset.
    """
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(
        numeric_df.corr(), annot=True, cmap='coolwarm',
        fmt='.2f', linewidths=0.5
    )
    plt.title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.show()


def statistical_analysis(df, col: str):
    """
    Computes statistical moments: mean, standard deviation, skewness, and kurtosis.

    :param df: Pandas DataFrame containing the dataset.
    :param col: Column name to analyze.
    :return: Tuple containing mean, standard deviation, skewness, and excess kurtosis.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the dataset by displaying summary statistics and correlations.

    :param df: Pandas DataFrame containing the dataset.
    :return: Cleaned and processed DataFrame.
    """
    print(df.describe())
    print(df.head())
    print(df.tail())

    numeric_df = df.select_dtypes(include=['number'])
    print(numeric_df.corr())

    return df


def writing(moments, col):
    """
    Prints the statistical moments and interprets skewness and kurtosis.

    :param moments: Tuple containing mean, stddev, skew, and kurtosis.
    :param col: Column name for which statistics are computed.
    """
    print(
        f'For the attribute {col}:\n'
        f'Mean = {moments[0]:.2f}, '
        f'Standard Deviation = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f}, '
        f'Excess Kurtosis = {moments[3]:.2f}.'
    )

    if moments[2] > 2:
        skewness_interpretation = 'highly right-skewed'
    elif moments[2] < -2:
        skewness_interpretation = 'highly left-skewed'
    elif moments[2] > 0:
        skewness_interpretation = 'slightly right-skewed'
    elif moments[2] < 0:
        skewness_interpretation = 'slightly left-skewed'
    else:
        skewness_interpretation = 'symmetrical'

    if moments[3] > 0:
        kurtosis_interpretation = 'leptokurtic (more peaked)'
    elif moments[3] < 0:
        kurtosis_interpretation = 'platykurtic (flatter)'
    else:
        kurtosis_interpretation = 'mesokurtic (normal distribution)'

    print(f'The data is {skewness_interpretation} and {kurtosis_interpretation}.')


def main():
    """
    Main function to load data, process it, generate plots, and analyze statistics.
    """
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'Churn Score'  # Column to analyze
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
