# Predicting Cyberbullying on Social Media

**Author:** Shivaraj  
**Project Type:** Machine Learning Classification  
**Dataset:** Cyberbullying Tweets Dataset  

## 📋 Project Overview

This project focuses on developing machine learning models to automatically detect and classify different types of cyberbullying in social media tweets. The system can identify various forms of cyberbullying including age-based, ethnicity-based, gender-based, religion-based, and other forms of cyberbullying.

## 🎯 Objectives

- Analyze cyberbullying patterns in social media tweets
- Develop and compare multiple machine learning models for cyberbullying detection
- Implement deep learning approaches using LSTM and BERT for improved accuracy
- Create a robust text preprocessing pipeline for social media data
- Build an interactive prediction system for real-time cyberbullying detection

## 📊 Dataset

The project uses a cyberbullying tweets dataset containing:
- **Features:** Tweet text content
- **Target:** Cyberbullying types (age, ethnicity, gender, religion, not_cyberbullying)
- **Size:** Balanced dataset across different cyberbullying categories
- **Format:** CSV file with tweet text and corresponding labels

## 🔧 Technologies Used

- **Python 3.x**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly, wordcloud
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **NLP:** NLTK, transformers (BERT)
- **Text Processing:** demoji, re, string

## 📁 Project Structure

```
cyberbullying_tweets/
│
├── Predicting_Cyberbullying_Shivaraj.ipynb    # Main notebook with complete analysis
├── cyberbully_data.csv                        # Processed dataset
├── cyberbullying_tweets (1).csv              # Original dataset
├── AI_Predicting Cyberbullying on Social Media (1).pdf  # Project documentation
└── README.md                                  # Project documentation
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cyberbullying_tweets
```

2. **Install required packages**
```bash
pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn tensorflow
pip install nltk transformers torch
pip install wordcloud demoji
```

3. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 📈 Methodology

### 1. Data Preprocessing
- **Text Cleaning:** Removal of hashtags, mentions, URLs, and special characters
- **Normalization:** Converting to lowercase, removing punctuation
- **Tokenization and Lemmatization:** Using NLTK WordNetLemmatizer
- **Stopword Removal:** Custom stopwords list for social media text
- **Emoji Handling:** Converting emojis to descriptive text

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of cyberbullying types
- Word frequency analysis across different categories
- Word clouds for visual representation
- Tweet length distribution analysis

### 3. Feature Engineering
- **TF-IDF Vectorization:** Converting text to numerical features
- **Word Embeddings:** For deep learning models
- **Sequence Padding:** For LSTM models

### 4. Model Development

#### Traditional Machine Learning Models:
- **Support Vector Machine (SVM)** with RBF kernel
- **Random Forest Classifier** with 100 estimators
- **Gradient Boosting Classifier**

#### Deep Learning Models:
- **LSTM (Long Short-Term Memory)** with embedding layer
- **Bidirectional LSTM** for improved context understanding
- **BERT (Bidirectional Encoder Representations from Transformers)** for state-of-the-art performance

## 📊 Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Support Vector Machine | ~85% | Good baseline performance |
| Random Forest | ~87% | Best traditional ML model |
| Gradient Boosting | ~86% | Consistent performance |
| LSTM | ~88% | Good for sequential data |
| Bidirectional LSTM | ~90% | Improved context understanding |
| BERT | ~92% | Best overall performance |

## 🔍 Key Features

### Text Preprocessing Pipeline
- Robust cleaning function handling social media specific content
- Custom stopwords for better noise reduction
- Emoji-to-text conversion for sentiment preservation

### Model Pipeline
- Scikit-learn pipeline for streamlined preprocessing and prediction
- Modular design for easy model swapping
- Integrated vectorization and classification

### Interactive Prediction System
- Real-time cyberbullying detection
- User-friendly input interface
- Multiple model options for comparison

## 💡 Usage

### Running the Complete Analysis
```python
# Open the Jupyter notebook
jupyter notebook Predicting_Cyberbullying_Shivaraj.ipynb
```

### Using the Trained Models
```python
# Load the pipeline model
import pickle
with open('model_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
tweet = ["Your sample tweet here"]
prediction = model.predict(tweet)
print(f"Predicted cyberbullying type: {prediction[0]}")
```

### BERT Model Predictions
```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load saved model
tokenizer = BertTokenizer.from_pretrained("cyberbullying_bert_tokenizer")
model = BertForSequenceClassification.from_pretrained("cyberbullying_bert_model")

# Predict
def predict_cyberbullying(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    return label_names[prediction.item()]
```

## 📝 Key Insights

1. **Balanced Dataset:** The dataset shows relatively equal distribution across cyberbullying types
2. **Word Patterns:** Different cyberbullying types show distinct vocabulary patterns
3. **Model Performance:** BERT significantly outperforms traditional ML models
4. **Text Length:** Most cyberbullying tweets are concise (under 50 words)

## 🔮 Future Enhancements

- **Real-time Detection:** Integration with social media APIs
- **Multilingual Support:** Extending to other languages
- **Severity Classification:** Adding severity levels to cyberbullying types
- **Ensemble Methods:** Combining multiple models for better accuracy
- **Explainable AI:** Adding model interpretability features

## 📚 References

- Natural Language Toolkit (NLTK)
- Scikit-learn Documentation
- TensorFlow/Keras Documentation
- Transformers Library (Hugging Face)
- BERT: Pre-training of Deep Bidirectional Transformers

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the preprocessing pipeline
- Adding new model architectures
- Enhancing the evaluation metrics
- Creating better visualizations

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

**Author:** Shivaraj  
**Project:** Infosys Springboard Project  
**Focus:** Machine Learning for Cyberbullying Detection  

---

*This project was developed as part of the Infosys Springboard program focusing on AI applications for social good.*