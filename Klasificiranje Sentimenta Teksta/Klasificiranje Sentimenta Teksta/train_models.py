import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_review(review):
    return ' '.join(word for word in review.split() if word.lower() not in stopwords.words('english'))


def visualize_wordcloud(reviews, sentiment):
    wordcloud = WordCloud(height=600, width=1000, max_font_size=100)
    plt.figure(figsize=(15, 12))
    plt.imshow(wordcloud.generate(reviews), interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for {sentiment} reviews')
    plt.show()


def prepare_data(df):
    X = df['review']
    Y = df['sentiment']
    vect = TfidfVectorizer()
    X = vect.fit_transform(df['review'])
    return X, Y, vect


def train_evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    display.plot()
    plt.show()

    precision = precision_score(y_test, y_pred, pos_label='positive')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label='positive')
    class_report = classification_report(y_test, y_pred)

    print("Precision Score:", precision)
    print("Accuracy Score:", accuracy)
    print("Recall Score:", recall)
    print("Classification Report:")
    print(class_report)
    return model

if __name__ == "__main__":
   
    df = load_data('IMDB Dataset.csv')
    
    df['review'] = df['review'].apply(clean_review)

    
    visualize_wordcloud(' '.join(df['review'][df['sentiment'] == 'negative'].astype(str)), 'negative')
    visualize_wordcloud(' '.join(df['review'][df['sentiment'] == 'positive'].astype(str)), 'positive')

    
    X, Y, vect = prepare_data(df)

   
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

   
    print("Logistic Regression Model:")
    lr_model = train_evaluate_model(LogisticRegression(), x_train, x_test, y_train, y_test)

    
    print("\nMultinomial Naive Bayes Model:")
    nb_model = train_evaluate_model(MultinomialNB(), x_train, x_test, y_train, y_test)

   
    print("\nLinear Support Vector Classification Model:")
    svc_model = train_evaluate_model(LinearSVC(), x_train, x_test, y_train, y_test)

    
    with open('models.pkl', 'wb') as f:
        pk.dump((vect, lr_model, nb_model, svc_model), f)
        
        
