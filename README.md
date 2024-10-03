# Hotel Review Sentiment Analysis

## Background

User reviews of hotel services on Trip Advisor have become a crucial source in the hospitality industry to deeply understand customer preferences and experiences. The information contained in these reviews not only includes the positive and negative aspects of hotel services but also provides a clear picture of the factors influencing user satisfaction.

## Problem Statement

The primary objective of this project is to develop a model that can automatically classify the sentiment of hotel reviews. This will enable efficient analysis of user feedback on a large scale, providing deep insights into the elements that most influence customer experiences at hotels and offering a useful tool for hotel management to respond to reviews more effectively.

## Objective

This project aims to perform data processing and preparation using Natural Language Processing (NLP) techniques, implement an Artificial Neural Network (ANN) to classify the sentiment of these reviews, and measure and explain the performance of the ANN model based on evaluation metrics such as accuracy and AUC. Thus, the developed model is expected to make a significant contribution to improving user management and experience in the hospitality industry.

## Tools
[<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />](https://pandas.pydata.org/)
[<img src="https://img.shields.io/badge/Seaborn-388E3C?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn" />](https://seaborn.pydata.org/)
[<img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" />](https://numpy.org/)
[<img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />](https://matplotlib.org/)
[<img src="https://img.shields.io/badge/Scikit%20learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />](https://scikit-learn.org/)
[<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />](https://www.tensorflow.org/)
[<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />](https://keras.io/)

## Model Building

LSTM (Long Short-Term Memory) was chosen because of its ability to handle long text data and overcome the vanishing gradient problem that often occurs in conventional RNNs (Recurrent Neural Networks). LSTM allows the model to remember long-term information, which is useful in analyzing sequential data like text.

The metrics chosen for model evaluation are accuracy to measure the overall accuracy of sentiment class predictions and AUC (Area Under the ROC Curve) to evaluate the model's ability to separate sentiment classes considering the true positive rate and false positive rate. This combination of metrics provides a comprehensive picture of the model's performance in text classification tasks, focusing on prediction accuracy and the ability to distinguish sentiment classes.

## Model Improvement Training

In the model improvement training, I added callbacks. The use of EarlyStopping in the model training process is crucial as this callback allows the model to stop training itself when there is no significant improvement in the monitored evaluation metric, such as validation loss. This not only prevents overfitting by stopping training in a timely manner but also improves training efficiency by saving computation.

## Model Evaluation

Based on the analysis of hotel reviews, the majority of reviews have a rating of 5, accounting for 44.2% of the total data, indicating high satisfaction. Positive categories (ratings 4 and 5) are the most dominant, while neutral reviews (rating 3) are relatively few compared to negative reviews (ratings 1 and 2). This indicates an imbalance in the data, where satisfaction tends to dominate, but criticism is also significant.

Positive reviews highlight customer satisfaction with hotel services and facilities without significant criticism. Neutral reviews, although tend to be longer, provide feedback for specific improvements. Negative reviews reflect customer dissatisfaction with various aspects of the hotel. In conclusion, to improve services and customer satisfaction, focus should be given to understanding and responding to both neutral and negative reviews, while maintaining high standards that support the dominant positive reviews.

The model improvement has good performance with an accuracy of around 97.39% and an AUC of 99.60% on training data. However, performance on validation data shows a decrease, indicating possible overfitting or further adjustments needed for model generalization.

It appears that the Positive class (rating 5) dominates with 44.2% of the total data, while the Neutral class (rating 3) is the least with only 328 samples (10.7%). This indicates a significant imbalance in the data distribution, where minority classes like Neutral may have less significant influence in model learning.

To improve model performance, strategies such as handling class imbalance (oversampling or undersampling), further hyperparameter tuning (e.g., dropout rate, learning rate), and exploring more complex models to capture subtler patterns in reviews can be considered.

Although the model performs well in classifying positive sentiment, there is room to improve the ability to recognize neutral and negative sentiments. Focusing on more in-depth review analysis and model adjustments can help improve accuracy and usefulness in sentiment analysis applications for hotel reviews.

By using this model to automatically analyze hotel reviews, businesses can be more responsive to customer feedback and complaints. This enables them to quickly identify and address potential issues, thereby enhancing overall customer satisfaction.

The model can improve operational efficiency by automating the review analysis process. This reduces the time and effort required to manually analyze reviews, allowing the team to focus on more strategic and effective improvement actions.

Deeper analysis of customer reviews can provide valuable insights for decision-making. Businesses can use this information to develop more effective marketing strategies, identify market trends, and enhance differentiation from competitors.

## Data

The dataset can be accessed at: [Trip Advisor Hotel Reviews on Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews/data).

## Deployment

You can access the live model via the following link: [TripAdvisor Sentiment Analysis on Hugging Face](https://huggingface.co/spaces/Gieorgie/Tripadvisior_Sentiment_Analysis).


