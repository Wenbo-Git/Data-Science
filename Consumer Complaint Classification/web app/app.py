from flask import Flask, render_template, request, redirect
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet, ranges
from nltk.corpus import stopwords
import inference
import math
import re

STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
MAX_NUM_COMPLAINTS = 5

app = Flask(__name__)
app.vars = {}
app.vars['logistic_regression'] = None

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = text.replace(r'\d+', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def reset_global_vars():
    app.vars['list_of_complaints'] = []
    app.vars['list_of_complaint_names'] = []
    app.vars['list_of_predictions'] = []
    app.vars['list_of_list_of_list_of_words'] = []
    app.vars['list_of_list_of_probabilities'] = []
    app.vars['error'] = None

@app.route('/')
def main():
    return redirect('/classify')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'GET':
        return render_template('classify.html')
    else:
        # Clear global variables except the model
        reset_global_vars()

        # Load complaints
        list_complaints_after_clean_up = []
        for i in range(MAX_NUM_COMPLAINTS):
            complaint_name = 'complaint{}'.format(i+1)
            complaint = request.form[complaint_name]
            if complaint:
                app.vars['list_of_complaints'].append(complaint)
                app.vars['list_of_complaint_names'].append(complaint_name)
                list_complaints_after_clean_up.append(clean_text(complaint))

        # Load model
        try:
            if app.vars['logistic_regression'] is None:
                app.vars['logistic_regression'] = joblib.load('logistic_regression.pkl')
            logistic_regressor = app.vars['logistic_regression']
        except Exception as e:
            app.vars['error'] = str(e)
            return redirect('/error')
        
        # Make Predictions
        if len(list_complaints_after_clean_up) > 0 and logistic_regressor is not None:
            try:
                predictions = []
                predictions = logistic_regressor.predict(list_complaints_after_clean_up)
                if len(predictions) > 0:
                    for i in range(len(predictions)):
                        app.vars['list_of_predictions'].append(predictions[i])
                        list_of_list_of_words, list_of_probabilities = \
                            inference.calculate_probability_for_each_class(list_complaints_after_clean_up[i], logistic_regressor)
                        app.vars['list_of_list_of_list_of_words'].append(list_of_list_of_words)
                        app.vars['list_of_list_of_probabilities'].append(list_of_probabilities)
                return redirect('/result')
            except Exception as e:
                app.vars['error'] = str(e)
                return redirect('/error')
                           
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        classes = app.vars["logistic_regression"].classes_
        list_of_complaints = app.vars['list_of_complaints']
        list_of_complaint_names = app.vars["list_of_complaint_names"]
        list_of_predictions = app.vars["list_of_predictions"]
        list_of_list_of_probabilities = app.vars["list_of_list_of_probabilities"]
        list_of_list_of_list_of_words = app.vars["list_of_list_of_list_of_words"]
        
        outputs = []
        for i in range(len(list_of_complaint_names)):
            complaint = list_of_complaints[i]
            complaint_name = list_of_complaint_names[i]
            prediction = list_of_predictions[i].lower()
            list_of_probabilities = list_of_list_of_probabilities[i]
            list_of_list_of_words = list_of_list_of_list_of_words[i]

            source = ColumnDataSource(dict(x = classes, y = list_of_probabilities))
            labels = LabelSet(x = 'x', y = 'y', text = 'y', source = source)
            p = figure(plot_width = 800, plot_height = 800, 
            x_axis_label = "Products",
            y_axis_label = "Probability",
            x_range = source.data["x"],
            y_range = ranges.Range1d(start = 0, end = 1))
            p.vbar(source = source, x = 'x', top = 'y', width = 0.5, color = "#CAB2D6")
            p.xaxis.major_label_orientation = math.pi/3
            p.title.text = "Probabilities of Products"
            p.title.align = "center"
            p.add_layout(labels)
            script, div = components(p)

            output = {"complaint":complaint,
                "complaint name":complaint_name,
                "prediction":prediction,
                "list of list of words":list_of_list_of_words,
                "script":script,
                "div":div}

            outputs.append(output)

        return render_template('result.html', outputs = outputs)
        
@app.route('/error', methods=['GET', 'POST'])
def error():
    if request.method == 'GET':
        return render_template('error.html', error = app.vars['error'])

if __name__ == '__main__':
    app.run(port=33507)
    #app.debug=True
    #app.run(host='0.0.0.0')