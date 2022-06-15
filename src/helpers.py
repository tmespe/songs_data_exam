import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, cross_val_score
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report


def identify_number_categories(df):
    """
    This definition can be used to identify the number of categories of every categorical feature 
    
    @param df DataFrame 
    
    @return a DataFrame with the categorical features and number of categories"""

    categorical_columns = list(df.select_dtypes(['object']).columns)

    cat_df = []

    for c in categorical_columns:
        cat_df.append({"categorical_feature": c,
                       "number_categories": len(df[c].value_counts(dropna = False))
                    })
        
    return pd.DataFrame(cat_df).sort_values(by = "number_categories", ascending = False)


def one_hot(df, categorical_cols):
    """
    This definition can be used to one hot encode categorical data
    
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    
    @return a DataFrame with one-hot encoding
    """
    
    for c in categorical_cols:
        dummies = pd.get_dummies(df[c], prefix=c)
        # dummies = pd.get_dummies(df[c], prefix=c, drop_first=True) - you can also remove one of the one hot encoded dummy vars
        df = pd.concat([df, dummies], axis=1)
        df.drop(c, axis = 1, inplace = True)
    
    return df


def identify_highly_correlated_features(df, correlation_threshold):
    """
    This definition can be used to identify highly correlated features
    
    @param df pandas DataFrame
    @param correlation_threshold int 
    
    @return a DataFrame with highly correlated features 
    """
    
    corr_matrix = df.corr().abs() # calculate the correlation matrix with 
    high_corr_var = np.where(corr_matrix >= correlation_threshold) # identify variables that have correlations above defined threshold
    high_corr_var = [(corr_matrix.index[x], corr_matrix.columns[y], round(corr_matrix.iloc[x, y], 2))
                         for x, y in zip(*high_corr_var) if x != y and x < y] # identify pairs of highly correlated variables
    
    if high_corr_var != []:
        high_corr_var_df = pd.DataFrame(high_corr_var).rename(columns = {0: 'corr_feature',
                                                                     1: 'drop_feature',
                                                                     2: 'corrrelation_values'})
        high_corr_var_df = high_corr_var_df.sort_values(by = 'corrrelation_values', ascending = False)
    else:
        high_corr_var_df = print("there are no pairs of correlations with that threshold")
        
    return high_corr_var_df


def identify_low_variance_features(df, std_threshold):
    """
    This definition can be used to identify features with low varaince
    
    @param df pandas DataFrame
    @param std_threshold int 
    
    @return a list of features that have low variance
    """
    
    std_df = pd.DataFrame(df.std()).rename(columns = {0: 'standard_deviation'})

    low_var_features = list(std_df[std_df['standard_deviation'] < std_threshold].index)

    print("number of low variance features:", len(low_var_features))
    print("low variance features:", low_var_features)
    
    return low_var_features


def identify_missing_data(df):
    """
    This function is used to identify missing data
    
    @param df pandas DataFrame
    
    @return a DataFrame with the percentage of missing data for every feature and the data types
    """
    
    percent_missing = df.isnull().mean()
    
    missing_value_df = pd.DataFrame(percent_missing).reset_index() # convert to DataFrame
    missing_value_df = missing_value_df.rename(columns = {"index" : "feature",
                                                                0 : "percent_missing"}) # rename columns

    missing_value_df = missing_value_df.sort_values(by = ['percent_missing'], ascending = False) # sort the values
    
    data_types_df = pd.DataFrame(df.dtypes).reset_index().rename(columns = {"index" : "feature",
                                                                0 : "data_type"}) # rename columns
    
    missing_value_df = missing_value_df.merge(data_types_df, on = "feature") # join the dataframe with datatype
    
    missing_value_df.percent_missing = round(missing_value_df.percent_missing*100, 2) # format the percent_missing
    
    return missing_value_df


# def feature_importance_plot(model, X_train, n):
#     """Plots feature importance - this only works for Decision Tree based Models"""
#     plt.figure(figsize=(8, 5)) # set figure size
#     feat_importances = pd.Series(model.feature_importances_,
#                                  index = X_train.columns)
#     feat_importances.nlargest(n).plot(kind = 'bar')
#     plt.title(f"Top {n} Features")
#     plt.xticks(rotation=45)
#     plt.show()
    


# Set up classification report
def evaluate_model(model, y_true, y_pred):
    print('Model: ', model)
    print(classification_report(y_true, y_pred))
    print('\n')

# Set up the function to plot the confusion matrix for all models


def plot_confusion_matrix(model, title, y_true, y_pred, encoder):
    plt.figure(figsize=(10, 8))
    xticks = encoder.inverse_transform(model.classes_)
    yticks = encoder.inverse_transform(model.classes_)
    cm = confusion_matrix(y_true, y_pred)
    norm_cm = cm / cm.sum(axis=1).reshape(-1, 1)
    sns.heatmap(norm_cm, annot=True, cmap='Blues', xticklabels=xticks, yticklabels=yticks,
                vmin=0, vmax=1)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# function to show 5 comparative values -> predicted vs. actual


def print_5(y_pred, y_test):
    print("first five predicted values:", y_pred[0:5])
    print("first five actual values:", list(y_test[0:5]))

# Print scores function


def print_scores(model_name, y_pred, y_test):
    # Evaluate the Decision Tree Model on Test Data
    # the evaluation metrics for the model on the test set
    print("Results of the Decision Tree Model:")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Score:", round(accuracy, 2))
    precision = precision_score(y_test, y_pred, average='macro')
    print("Precision Score:", round(precision, 2))
    recall = recall_score(y_test, y_pred, average='macro')
    print("Recall Score:", round(recall, 2))
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score:", round(f1, 2))
    return pd.DataFrame(index=["Accuracy", "Precision", "Recall", "F1"],
                           data=[accuracy, precision, recall, f1], columns=[model_name])

def feature_importance_plot(model, x_train, n=10):
    """Plots feature importance - this only works for Decision Tree based Models"""
    # Select top_n features to plot
    feat_importances = pd.Series(model.feature_importances_,
                                 index=x_train.columns).nlargest(n)
    fig = px.bar(feat_importances, color=feat_importances.index,
                 labels={"value": "Importance", "index": "Feature"},
                 title="Feature importances")
    fig.update_layout(showlegend=False) #  Hide unecessary legend
    return fig

def learning_curve_plot(model, X_train, y_train, scoring):
    """Plots learning curves for model validation
    
    @param models - list of models we are interested in evaluating
    @param X_train - the training features
    @param y_train - the target
    
    @returns a plot of the learning curve
    """
    
    plt.figure(figsize=(5, 5)) # set figure size
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        # Number of folds in cross-validation
        cv = 5,
        # Evaluation metric
        scoring = scoring,
        # Use all computer cores
        n_jobs = -1,
        shuffle = True,
        # 5 different sizes of the training set
        train_sizes = np.linspace(0.01, 1.0, 5))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color = "#111111", label = "Training score")
    plt.plot(train_sizes, test_mean, color = "#111111", label = "Cross-validation score")

    # Draw bands
    # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color = "#DDDDDD")
    # plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color = "#DDDDDD")

    # Create plot
    plt.title("Learning Curves")
    plt.xlabel("Training Set Size"), plt.ylabel("Error"), plt.legend(loc = "best")
    plt.tight_layout()
    

    plt.show()