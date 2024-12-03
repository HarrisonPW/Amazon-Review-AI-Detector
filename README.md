

## Open Source Packages

### Libraries Used

1. **[Scikit-learn](https://scikit-learn.org/stable/)**
   - Description: Machine learning library for Python. Used for Logistic Regression and Multinomial Naive Bayes.
   - Language: Python.

2. **[Hugging Face Transformers](https://huggingface.co/docs/transformers/)**
   - Description: Library for state-of-the-art natural language processing models like GPT-2.
   - Language: Python.

3. **[Optuna](https://optuna.org/)**
   - Description: Framework for hyperparameter optimization.
   - Language: Python.

4. **[Pandas](https://pandas.pydata.org/)**
   - Description: Data manipulation and analysis library.
   - Language: Python.

5. **[Matplotlib](https://matplotlib.org/)**
   - Description: Library for creating static, animated, and interactive visualizations.
   - Language: Python.

6. **[NumPy](https://numpy.org/)**
Description: Fundamental package for numerical computation in Python, providing support for arrays and matrices.
Language: Python.

7. **[Seaborn](https://seaborn.pydata.org/)**
Description: Statistical data visualization library based on Matplotlib.
Language: Python.

8. **[PyTorch](https://pytorch.org/)**
Description: Deep learning framework providing tensor computation and automatic differentiation.
Language: Python.

9. **[NLTK (Natural Language Toolkit)](https://www.nltk.org/)**
Description: Library for processing and analyzing human language data (natural language processing).
Language: Python.

10. **[TQDM](https://tqdm.github.io/)**
Description: Library for creating progress bars in Python.
Language: Python.

11. **[Logging](https://docs.python.org/3/library/logging.html)**
Description: Built-in Python library for generating log messages.
Language: Python.

12. **[JSON](https://docs.python.org/3/library/json.html)**
Description: Built-in Python module for parsing and creating JSON data.
Language: Python.

13. **[Time](https://docs.python.org/3/library/time.html)**
Description: Built-in Python module for handling time-related functions.
Language: Python.

---

## Datasets

### Amazon Fake/Real Review Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/naveedhn/amazon-product-review-spam-and-non-spam/data)
- **Categories**: Cell Phones and Accessories, Clothing, Home and Kitchen, Sports, and Toys.
- **Size**: 100,000 data points (balanced dataset: 50% real and 50% fake reviews).
- **Preprocessing**:
  - Text cleaning (removing punctuations, stopwords, and extra spaces).
  - Train-Test Split: 80% training, 20% testing.

---

## Performance Measurement Tools

1. **Confusion Matrix**
   - Evaluates model prediction performance.
   - Highlights Type I and Type II errors.

2. **ROC and Precision-Recall Curves**
   - Visualization of performance metrics over different thresholds.

3. **Training and Evaluation Metrics**
   - Accuracy, Precision, Recall, F1 Score across epochs.
  
---

## Models

### **[Logistic Regression Model](https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html)**

**Overview**:  
Logistic Regression is a classic statistical method used for binary classification tasks. In this project, it is employed to classify Amazon reviews as either **spam** (fake) or **non-spam** (real). Despite its simplicity, Logistic Regression is effective for tasks where feature relationships are linear or near-linear.

**Key Features**:
- **Simplicity and Interpretability**: The model provides straightforward results and coefficients that explain the relationship between features and output.
- **Speed**: Training is computationally efficient, even on large datasets.
- **Binary Classification**: Suitable for a balanced dataset like the one used here.

**Training Details**:
- **Dataset Split**: 80% training, 20% testing.
- **Preprocessing**: 
  - Text cleaning (lowercasing, stopword removal, punctuation removal).
  - Features extracted using TF-IDF or Count Vectorization.
- **Metrics Evaluated**:
  - Accuracy, Precision, Recall, and F1 Score.
  - ROC and Confusion Matrix.

**Use Case**:  
Logistic Regression works best when computational resources are limited, or a fast and interpretable model is preferred.

---

### **[GPT-2 Model](https://huggingface.co/openai-community/gpt2)**

**Overview**:  
GPT-2 (Generative Pre-trained Transformer 2) is a deep learning model developed by OpenAI for natural language understanding and generation. In this project, GPT-2 is fine-tuned for **text classification**, specifically detecting fake and real reviews by analyzing patterns, semantics, and linguistic nuances.

**Key Features**:
- **Contextual Understanding**: GPT-2 excels in understanding the context and generating human-like text.
- **Fine-tuned for Specific Tasks**: By retraining on a labeled dataset, GPT-2 learns to distinguish spam reviews from genuine ones.
- **Adaptability**: GPT-2 can be used for text classification, summarization, and other NLP tasks.

**Training Details**:
- **Dataset Split**: 80% training, 20% testing.
- **Fine-tuning Parameters**:
  - **Epochs**: Adjusted to optimize performance.
  - **Batch Size**: Fine-tuned for resource optimization.
  - **Learning Rate**: Optimized using Optuna.
- **Performance Metrics**:
  - Accuracy, Precision, Recall, and F1 Score tracked across epochs.
  - Training Loss monitored for optimization.
  
**Advanced Features**:
- Uses **transformer architecture**, which includes multi-head attention and positional encodings.
- Capable of handling long-range dependencies in text, making it ideal for nuanced tasks like fake review detection.

**Use Case**:  
GPT-2 is suitable for tasks requiring a high degree of text understanding or tasks where leveraging semantic and contextual information significantly improves performance.

---

### **[Gemini 1.5 API Model](https://ai.google.dev/gemini-api/docs/model-tuning)**  

**Overview**:  
The **Gemini 1.5 API** is a cutting-edge large language model (LLM) optimized for a wide range of natural language processing tasks. In this project, it was fine-tuned to classify Amazon reviews, leveraging its advanced semantic understanding and text classification capabilities.  

**Key Features**:  
- **Pre-trained for Versatility**: Gemini 1.5 comes pre-trained on extensive datasets, making it adaptable for tasks such as classification, summarization, and sentiment analysis.  
- **High Accuracy in Text Understanding**: Its ability to process complex language patterns contributes to robust classification results.  
- **Cloud-based Scalability**: As an API, it supports seamless integration and scales effortlessly for production environments.  

**Training Details**:  
- **Sample Size**: 100,000 samples in total. 80,000 samples in the training dataset. 20,000 samples in the testing dataset.
- **Input**: TextReview (removing stopwords)
- **Output**: Class (0 or 1)
- **Dataset Split**: 80% training, 20% testing.  
- **Fine-tuning Parameters**:  
  - Optimized using Optuna.
  - **Epochs**: Set to the maximum (2) under the constraint of Google AI Studio.  
  - **Batch Size**: Tuned for resource efficiency.  
  - **Learning Rate**: Adjusted to minimize overfitting and maximize generalization.  
- **Metrics Evaluated**:  
  - Accuracy, Precision, Recall, F1 Score, and confusion matrix.  
  - Reduction in cross-entropy loss (<= 1) achieved during training.  

**Advanced Features**:  
- **Transformer and MoE Architecture**: Incorporates self-attention and positional encodings for understanding text at both a local and global level.  
- **API-Driven Deployment**: Accessible via a secure, scalable API endpoint, enabling real-time inference for review classification tasks.  
- **Real-Time Integration**: Integrated with other models in the system for collaborative inference, enhancing overall system accuracy.  

**Use Case**:  
Gemini 1.5 API is ideal for applications requiring a balance between high accuracy and deployment scalability, such as real-time content moderation, sentiment analysis, and review authenticity verification.  

---

### Model Comparison

| Feature                | Logistic Regression                        | GPT-2                                   | Gemini 1.5 API                         |
|------------------------|--------------------------------------------|-----------------------------------------|-----------------------------------------|
| **Complexity**         | Low                                       | High                                    | High                                    |
| **Training Time**      | ~28 seconds for 100 epochs                | Hours per epoch (GPU-accelerated)       | Cloud-based, offloaded computation      |
| **Interpretability**   | High (coefficients are interpretable)      | Low (black-box neural network)          | Moderate (API abstraction)             |
| **Resource Needs**     | Minimal (CPU sufficient)                  | High (requires GPU for efficient use)   | Moderate (API-driven scaling)          |
| **Accuracy**           | Moderate (good for linear data)           | High (excels in capturing nuanced data) | High (API-trained on extensive datasets) |
| **Suitability**        | Simple, fast tasks with limited features   | Complex, semantic-rich text tasks       | Versatile, real-time NLP tasks          |  


## To run this project on your computer, you can follow these steps:

### 1. Clone or Download the Project

On GitHub, clone or download this project to your local environment:

```bash
git clone <repository_url>
cd <repository_name>
```
To run the backend, go to the folder `Amazon-Review-AI-Detector`.
To run the frontend, go to the folder `amazon-spam-detector-frontend`.

### 2. Run on Docker

Ensure Docker Desktop is Running. 

Run the following command:

```bash
sudo docker-compose up --build -d
```
OR (for Windows)

```bash
sudo docker-compose up --build -d
```


### 3. Start the Development Server

After running this command, you should see output in the terminal similar to:

```
  VITE v5.x.x  ready in xx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 4. Access the Project in Your Browser

Open your browser and go to the local address shown in the output (usually `http://localhost:5173`). You should be able to see the frontend project running on your local server.

### Explanation of Other Files

- **Dockerfile and docker-compose.yml**: If you want to run this project in a Docker container, you can use these files. Run `docker-compose up` to start the container.
- **tsconfig.json**: TypeScript configuration file, which defines the TypeScript compilation options.
- **eslint.config.js**: ESLint configuration file, used for code style and quality checking.
