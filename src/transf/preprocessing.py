"""
Module for preprocessing text data for classification
"""

import re
from typing import List, Dict, Tuple, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def handle_negations(text: str, negations: List[str]) -> str:
    """
    Detect and transform negated phrases in the text.
    
    Args:
    text (str): The input text.
    negations (List[str]): List of negation words.
    
    Returns:
    str: The text with negated phrases transformed.
    """
    words = text.split()  # Tokenize the text
    transformed_words = []
    negate = False

    i = 0
    while i < len(words):
        word = words[i]
        # If negation detected, append "not" to the following word
        if word in negations and i + 1 < len(words):
            transformed_words.append(f"not_{words[i + 1]}")
            negate = True
            i += 1  # Skip the next word as it's combined with negation
        elif negate:
            transformed_words.append(f"not_{word}")
            negate = False
        else:
            transformed_words.append(word)
        i += 1
    return ' '.join(transformed_words)

def handle_repeated_characters(word: str) -> str:
    """
    Handle repeated characters in a word (e.g., "soooo" -> "soo").
    
    Args:
    word (str): The input word.
    
    Returns:
    str: The word with repeated characters handled.
    """
    return re.sub(r'(.)\1+', r'\1\1', word)

def clean_text(text: str, negations: List[str]) -> str:
    """
    Preprocess the text for sentiment analysis.
    
    Args:
    text (str): The input text.
    negations (List[str]): List of negation words.
    
    Returns:
    str: The cleaned and preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove all punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Split the text into words
    words = text.split()
    
    # Remove stopwords, but keep negations and important words like "but", "very", etc.
    words = [word for word in words if word not in stop_words or word in negations]
    
    # Handle negations by concatenating them with the following word
    neg_handled_text = handle_negations(' '.join(words), negations)
    
    # Tokenize the processed text after handling negations
    words_processed = neg_handled_text.split()
    
    # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_processed]
    
    # Handle repeated characters
    lemmatized_words = [handle_repeated_characters(word) for word in lemmatized_words]
    
    # Join the cleaned words back into a string
    cleaned_text = ' '.join(lemmatized_words)
    
    return cleaned_text


def plot_confusion_matrix(conf_matrix: Any, figsize: tuple = (8, 6), cmap: str = 'viridis') -> None:
    """
    Plot the confusion matrix using Seaborn.

    Args:
    conf_matrix (Any): The confusion matrix to be plotted.
    figsize (tuple, optional): The size of the figure. Defaults to (8, 6).
    cmap (str, optional): The colormap to be used. Defaults to 'viridis'.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()



def plot_roc_curve(y_true: List[int], y_pred_prob: List[float], width: int = 1000, height: int = 400) -> None:
    """
    Plot the ROC curve using Plotly.

    Args:
    y_true (List[int]): True binary labels.
    y_pred_prob (List[float]): Target scores, can either be probability estimates of the positive class or confidence values.
    width (int, optional): The width of the plot. Defaults to 1000.
    height (int, optional): The height of the plot. Defaults to 400.
    """
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Create the ROC curve plot
    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC curve (area = {roc_auc:.2f})',
        line=dict(color='red', width=2)
    ))

    # Add a diagonal line representing a random classifier
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # Update layout for the ROC curve plot
    fig_roc.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=width,  # Set the width of the figure
        height=height,  # Set the height of the figure
        showlegend=True,
        legend=dict(x=1.05, y=1, orientation='v'),  # Place legend on the right
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Remove background
        paper_bgcolor='rgba(0,0,0,0)',  # Remove background
        font=dict(color='black')  # Set font color to black
    )

    # Show the ROC curve plot
    fig_roc.show()