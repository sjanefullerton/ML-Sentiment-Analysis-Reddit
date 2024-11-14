# **Reddit Sentiment Analysis using API Web Scraper and Fine-Tuned Machine Learning Models**

### Overview:
This project involves collecting Reddit data related to **Hurricane Helene** using an API web scraper built with **PRAW**, followed by preprocessing the data and training **DistilBERT** models for sentiment analysis. The sentiment models were fine-tuned on manually labeled data and then used to automatically label the remainder of the dataset. This approach enabled efficient sentiment classification of posts and comments, aiding in the analysis of public opinion around natural disasters.

### Key Steps:
1. **Data Collection via Web Scraping**:
   - Built an API-based web scraper using **PRAW** to collect Reddit posts, comments, and titles related to **Hurricane Helene**.
   - Extracted relevant textual data from Reddit for further analysis.

2. **Data Preprocessing**:
   - Cleaned and preprocessed the scraped data using **Pandas**, including handling missing values and formatting issues.
   - Manually labeled 200 observations with sentiment labels (positive, negative, neutral) for use in model training.

3. **Model Fine-Tuning**:
   - Fine-tuned **DistilBERT** models using the **Transformers** library for sentiment classification on three types of text:
     - **Title Sentiment**
     - **Post Content Sentiment**
     - **Comment Sentiment**
   - Each model was trained on the manually labeled data, ensuring the models could classify sentiment based on the respective text types.

4. **Automated Labeling**:
   - After fine-tuning, the trained models were used to automatically label the sentiment of the remaining scraped data.
   - This process significantly streamlined sentiment classification for the entire dataset.

5. **Model Evaluation**:
   - Evaluated model performance using metrics such as **loss**, **accuracy**, and **runtime** to ensure that the models were working effectively and efficiently.
   - Fine-tuned hyperparameters such as **learning rate**, **batch size**, and **epochs** to optimize the models’ accuracy.

### Technologies Used:
- **Python**: The primary programming language for scraping, preprocessing, and model training.
- **Pandas**: For data manipulation, cleaning, and preprocessing.
- **PRAW**: For web scraping data from Reddit via the Reddit API.
- **Transformers**: For using **DistilBERT** to fine-tune models for sentiment analysis.
- **Hugging Face**: To access pre-trained models and fine-tune them on the labeled dataset.
- **Matplotlib/Seaborn**: For visualizing model performance and understanding data patterns.

### Results:
- Successfully trained and fine-tuned three sentiment models that accurately classified Reddit data related to Hurricane Helene.
- Automated the labeling of thousands of additional posts and comments with high accuracy, streamlining sentiment analysis for the entire dataset.
  
### Next Steps/Future Work:
- Expand the dataset by scraping data related to other natural disasters or major events.
- Enhance model performance by incorporating more labeled data and experimenting with additional fine-tuning techniques.
- Develop a web interface or application for real-time sentiment analysis on Reddit data.
