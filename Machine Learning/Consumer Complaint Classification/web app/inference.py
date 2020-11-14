import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

def softmax(tfidf, dic_weights, dic_bias, c):
    temp1 = np.exp(tfidf.dot(dic_weights[c]) + dic_bias[c])
    numerator = temp1/(1 + temp1) #prevent numerical overflow
    denominator = 0
    for key, weight in dic_weights.items():
        temp2 = np.exp(tfidf.dot(weight) + dic_bias[key])
        denominator = denominator + temp2/(1 + temp2)
    prob = numerator/denominator
    return prob

def get_model_params(model):
    classes = model.named_steps["classifier"].classes_
    weight_matrix = model.named_steps["classifier"].coef_
    bias_vector = model.named_steps["classifier"].intercept_
    dic_weights = dict(zip(classes, weight_matrix))
    dic_bias = dict(zip(classes, bias_vector))
    return dic_weights, dic_bias

def get_tfidf_features_from_text(model, text):
    ls_text = []
    ls_text.append(text)
    word_counts = model.named_steps["bow"].transform(ls_text).toarray()
    tfidf = model.named_steps["tfidf"].transform(word_counts).toarray()
    return tfidf

def get_N_most_relevant_words(c, model, text, N):
    feature_names = model.named_steps["bow"].get_feature_names()
    dic_weights, dic_bias = get_model_params(model)
    tfidf = get_tfidf_features_from_text(model, text)
    result = np.multiply(tfidf, dic_weights[c])
    
    # get N most relevant words
    N_relevant_words = []
    sorted_index = np.argsort(-1 * result[0]) # sort in descending order
    for i in range(N):
        N_relevant_words.append(feature_names[sorted_index[i]])
    
    # filter out unseen words
    N_relevant_filtered_words = []
    for word in N_relevant_words:
        if word.lower() in text.lower():
            N_relevant_filtered_words.append(word)
    
    # Compute probability
    prob = softmax(tfidf, dic_weights, dic_bias, c)
        
    return N_relevant_filtered_words, prob

def calculate_probability_for_each_class(complaint, model, num_of_relevant_words = 5):
    list_of_probabilities = []
    list_of_list_of_words = []
    if complaint is not None:
        for c in model.classes_:
            ls_words, prob = get_N_most_relevant_words(c, model, complaint, num_of_relevant_words)
            list_of_probabilities.append(prob[0])
            list_of_list_of_words.append(ls_words)
    return list_of_list_of_words, list_of_probabilities



    