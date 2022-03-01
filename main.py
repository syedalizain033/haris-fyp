
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag, map_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pypyodbc as odbc # pip install pypyodbc
from IPython.display import HTML
import configparser
from io import BytesIO
import base64
import os
import tweepy as tw
import re
from flask import Flask, render_template
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from flask import Flask, render_template , request , redirect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.linear_model import LogisticRegression
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns

app = Flask(__name__)

dataset = pd.read_csv('FinalLabelitS.csv')

featuress = dataset[['User_ID' , 'User_Name' , 'location' , 'Retweet_Count', 'Followers','Following','Account_Likes','Listed_Count','Account_Type' ,'Created_at','Verified','Event']]
text = dataset[['User_Name','Tweet_Text', 'Retweet_Count' , 'FINAL ']]

features = dataset[['Retweet_Count', 'Followers', 'Following', 'Account_Likes',
       'Listed_Count', 'Account_Type']]
X = np.asarray(features)
y = np.asarray(dataset['FINAL '])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/index')
def home():
    return render_template('index.html')



@app.route('/fetchdata')
def fetchdata():

    return render_template('fetchfeatures.html', tables=[featuress.to_html(classes='data')], titles=featuress.columns.values)


@app.route('/fetchtweet')
def fetchtweet():
    return render_template('fetchtweet.html', tables=[text.to_html(classes='data')], titles=text.columns.values)


@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/displayform')
def displayform():
    return render_template('sentiment.html')


@app.route('/eda')
def eda():
    return render_template('EDA.html')


@app.route('/model')
def model():
    names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
             "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
             "Naive_Bayes", "Logistic_Regression" ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="poly", degree=3, C=0.025),
        SVC(kernel="rbf", C=1, gamma=2),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
        DecisionTreeClassifier(max_depth=5),
        ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
        RandomForestClassifier(max_depth=5, n_estimators=100),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(n_estimators=100),
        GaussianNB(),
        LogisticRegression(random_state=42)]

    scores = []
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)

    df = pd.DataFrame()
    df['name'] = names
    df['score'] = scores

    sns.set(style="whitegrid")
    ax = sns.barplot(y="name", x="score", data=df)

    sns.set()
    sns_plot = sns.barplot(y="name", x="score", data=df)
    figure = sns_plot.get_figure()
    # sns.plt.show()
    img = BytesIO()
    # plotting the confusion matrix
    figure.set_size_inches(10, 4)  # set figure's size manually to your full screen (32x18)
    plt.savefig(img, bbox_inches='tight', dpi=100)  # bbox_inches removes extra white spaces
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('showdataframe.html' ,tables=[df.to_html(classes='data')], titles=df.columns.values ,plot_url=plot_url)







@app.route('/sentimentt', methods = ["GET" , "POST"])
def sentimentt():
    if request.method == "POST" :
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score["neg"] != 0:
            message ="Negative🙁☹"
            return render_template('msg.html' , value = message  )
        else:
            message = "Positive😀😃"
            return render_template('msg.html' ,value = message ) 


@app.route('/svm')
def svm():
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    acu = metrics.accuracy_score(y_test, y_pred)
    img = BytesIO()
    # plotting the confusion matrix
    plot_confusion_matrix(classifier, X_test, y_test)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 4)  # set figure's size manually to your full screen (32x18)
    plt.savefig(img, bbox_inches='tight', dpi=100)  # bbox_inches removes extra white spaces
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    report = classification_report(y_test, y_pred)
    return render_template('svmresult.html' , value = acu ,plot_url=plot_url , report = str(report) )

@app.route('/randomforest')
def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=150)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Import scikit-learn metrics module for accuracy calculation
    # Model Accuracy, how often is the classifier correct?
    acuu  = metrics.accuracy_score(y_test, y_pred)
    img = BytesIO()
    # plotting the confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 4)  # set figure's size manually to your full screen (32x18)
    plt.savefig(img, bbox_inches='tight', dpi=100)  # bbox_inches removes extra white spaces
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    report = classification_report(y_test, y_pred)
    return render_template('rfresult.html', value=acuu, plot_url=plot_url, report=str(report))


@app.route('/logisticregression')
def logisticregression():
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    acuuu = metrics.accuracy_score(y_test, y_pred)
    img = BytesIO()
    # plotting the confusion matrix
    plot_confusion_matrix(logreg, X_test, y_test)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 4)  # set figure's size manually to your full screen (32x18)
    plt.savefig(img, bbox_inches='tight', dpi=100)  # bbox_inches removes extra white spaces
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    report = classification_report(y_test, y_pred)
    return render_template('lr.html', value=acuuu, plot_url=plot_url, report=str(report))



@app.route('/naivebayes')
def naivebayes():
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    img = BytesIO()
    # plotting the confusion matrix
    plot_confusion_matrix(classifier, X_test, y_test)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 4)  # set figure's size manually to your full screen (32x18)
    plt.savefig(img, bbox_inches='tight', dpi=100)  # bbox_inches removes extra white spaces
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    report = classification_report(y_test, y_pred)
    return render_template('nb.html', value=ac, plot_url=plot_url, report=str(report))



@app.route('/knn')
def knn():
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    # Calculate the accuracy of the model
    #print(knn.score(X_test, y_test))

    y_pred = knn.predict(X_test)
    ac = accuracy_score(y_test, y_pred)
    img = BytesIO()
    # plotting the confusion matrix
    plot_confusion_matrix(knn, X_test, y_test)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 4)  # set figure's size manually to your full screen (32x18)
    plt.savefig(img, bbox_inches='tight', dpi=100)  # bbox_inches removes extra white spaces
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    report = classification_report(y_test, y_pred)
    return render_template('knn.html', value=ac, plot_url=plot_url, report=str(report))



@app.route('/data')
def data():
    return render_template('form.html')



@app.route('/twitterapi' , methods=['POST','GET'])
def twitterapi():
    if request.method == 'POST':
        form = request.form
        search_words = request.form['name']
        date_since = request.form['date']
        X = int(request.form['limit'])


    config = configparser.RawConfigParser()
    config.read('credentials.ini')
    print(config.sections());
    accesstoken = config.get('twitter', 'accesstoken')
    accesstokensecret = config.get('twitter', 'accesstokensecret')
    apikey = config.get('twitter', 'apikey')
    apisecretkey = config.get('twitter', 'apisecretkey')

    auth = tw.OAuthHandler(apikey, apisecretkey)
    auth.set_access_token(accesstoken, accesstokensecret)
    api = tw.API(auth, wait_on_rate_limit=True)

    tweets = tw.Cursor(api.search, q=search_words, lang="en", since=date_since).items(X)

    tweets

    tweet_details = [[tweet.user.id, tweet.user.name, tweet.text, tweet.user.location, tweet.retweet_count,
                      tweet.user.followers_count, tweet.user.friends_count, tweet.user.favourites_count,
                      tweet.user.listed_count, tweet.user.protected, tweet.user.created_at, tweet.user.verified] for
                     tweet in tweets]

    tweet_df = pd.DataFrame(data=tweet_details,
                            columns=["User_ID", "User_Name", "Text", "location", "Retweet_Count", "Followers",
                                     "Following", "Account_Likes", "Listed_Count", "Account_Type", "Created_at",
                                     "Verified"])

    def clean_tweets(text):
        text = re.sub("RT @[\w]*:", "", text)
        text = re.sub("@[\w]*", "", text)
        text = re.sub("https?://[A-Za-z0-9./]*", "", text)
        text = re.sub("\n", "", text)
        return text

    tweet_df['Text'] = tweet_df['Text'].apply(lambda x: clean_tweets(x))
    tweet_df['Event'] = search_words

    tweet_df['location'].replace('', "None", inplace=True)
    tweet_df.head(5)
    tweet_df.to_csv(search_words + '_Full.csv', index=False, header=True, mode='a')
    #print("Data Has been Generating succesfully and CSV is in your system")
    return render_template('output.html')

#if __name__ == '__main__':
#    app.run(debug=True)



