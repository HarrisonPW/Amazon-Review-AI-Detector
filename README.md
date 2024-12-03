

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
