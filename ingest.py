#Topic Modeling Dependencies
import gensim, spacy
from sys import argv
from nltk.corpus import stopwords
import pandas as pd
import os

#Visualization Dependencies
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


#####
#Various parameters for tuning
#####


exclude_blank_lines = True

#determines whether dataframes are written
write_csv = True
output_directory = 'csv'

#If enabled, the number of topics set in file_topics_dict is used instead of settings below
create_visualization = True


num_topics = 4
#if enabled, iterates through the provided range and outputs all csvs
#Overrides parameter above
use_range_of_topics = False
num_topics_range = range(2,11)


filter_with_stop_words = True
filter_with_allowed_pos_tags = True
allowed_pos_tags = ['NOUN','ADJ','VERB','ADV']

#Used to control the number of topics found and their labels of Keywords
#Keywords are used in visualizations
file_topics_dict = {'q1.txt':10,'q3.txt':7,'q4.txt':2,'q5.txt':2,'q6.txt':8,'q7.txt':3,'q8.txt':10}
topic_dict = {
'q1.txt':['Exposure Data\nand Tracking','None','Avoid Contact With\nSick Individuals','Desire for Privacy','Sickness/Infection','Family Members','Economy','Would Consider','Family/Personal','Required'],
'q3.txt':['None','Private Insurance and Health Care Providers','Government','Hackers and Police','Family','Parents and Significant Others','Unsure'],
'q4.txt':['Data Used for Invasive Tracking','No Thoughts'],
'q5.txt':['None','Information'],
'q6.txt':['None','Never','Trust','Mandate','Collected Information','Negative Feelings\ntowards Concept','Data Safety and Privacy','Information Security\nand Management'],
'q7.txt':['Government','Data and Reputability','Proof It Works'],
'q8.txt':['Guarantee of Privacy\nand Accuracy','Occupational Need','Health Related Issues','Employer','Medical Information','Provider','Family','Intrusiveness','Information Security','Discriminination']
}

#####
#Helper Functions
#####
def clean_response(response, filter_with_stop_words = filter_with_stop_words):
    #Replace these characters with spaces
    exclude_list = ['.','"','/','\\','\n']

    #Remove these characters
    remove_list = ['\'']

    for char in exclude_list:
        response = response.replace(char,' ')

    for char in remove_list:
        response = response.replace(char,'')

    #Returns a list of tokens
    #Processed with lowercasing and deaccent, and filtered through stop stop words
    comment_list = list(gensim.utils.simple_preprocess(response, deacc=True))

    if filter_with_stop_words:
        stop_words = stopwords.words('english')
        comment_tokens = [word for word in comment_list if word not in stop_words]
    else:
        comment_tokens = [word for word in comment_list]


    return comment_tokens


def get_lines(file):
    lines = open(file).readlines()
    line_list = []

    for line in lines:
        if exclude_blank_lines and line=='':
            continue
        line_list.append(line)

    response_token_list = [clean_response(response) for response in line_list]

    return line_list, response_token_list

#Taken from Section 4 of https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
def build_models(response_token_list, min_count = 1, threshold = 1, allowed_postags = allowed_pos_tags, filter_with_stop_words = filter_with_stop_words, filter_with_allowed_pos_tags = filter_with_allowed_pos_tags):
    #Set up models to process words to create bigrams and trigrams
    bigram = gensim.models.Phrases(response_token_list, min_count = min_count, threshold = threshold)
    trigram = gensim.models.Phrases(bigram[response_token_list], threshold = threshold)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)

    bigram_texts = [bigram_model[response] for response in response_token_list]
    trigram_texts = [trigram_model[bigram_model[response]] for response in response_token_list]

    nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])
    processed_trigrams = []

    for response in trigram_texts:
        if filter_with_allowed_pos_tags:
            processed_trigrams.append([token.lemma_ for token in nlp(" ".join(response)) if token.pos_ in allowed_postags])
        else:
            processed_trigrams.append([token.lemma_ for token in nlp(" ".join(response))])


    if filter_with_stop_words:
        stop_words = stopwords.words('english')
        processed_trigrams = [[word for word in gensim.utils.simple_preprocess(str(response)) if word not in stop_words] for response in processed_trigrams]
    else:
        processed_trigrams = [[word for word in gensim.utils.simple_preprocess(str(response))] for response in processed_trigrams]

    return processed_trigrams

def create_topic_model(processed_output, num_topics, create_visualization, topic_dict, filename):
    id_word_map = gensim.corpora.Dictionary(processed_output)
    corpus = [id_word_map.doc2bow(comment) for comment in processed_output]
    lda = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id_word_map, num_topics = num_topics, random_state = 100, update_every = 1, chunksize = 10, passes = 10, alpha = 'asymmetric', iterations = 100, per_word_topics = True)

    #Used as a measurement in an attempt to gauge the optimal number of topics
    coherence_score = gensim.models.coherencemodel.CoherenceModel(model = lda, texts = processed_output, dictionary = id_word_map, coherence='u_mass').get_coherence()
    print('On {} topics, coherence of {}'.format(num_topics, coherence_score))

    if create_visualization:
        tSNE_visualization(lda,corpus, num_topics, topic_dict, filename)
    return get_dominant_topics(lda,corpus,processed_output)


#Used directly from https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#4.-Build-the-Bigram,-Trigram-Models-and-Lemmatize
def get_dominant_topics(lda_model, corpus, texts):


    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    return df_dominant_topic

#Outputs the lda model into a Pandas dataframe for easy manipulation
#Used directly from https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#4.-Build-the-Bigram,-Trigram-Models-and-Lemmatize
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

#taken from section 13 of
#https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models
def tSNE_visualization(lda_model, corpus, num_topics, topic_dict, filename):

    question_dict = {
    'q1.txt':'Please specify other reasons, if any,\n for why you might personally utilize a track-and-trace app.',
    'q2.txt':'Please describe the conditions under which\n you would use this status app.',
    'q3.txt':'Is there a group we did not list that you would trust?\n If so, who would that be?',
    'q4.txt':'Tell us any of your thoughts about using a smart phone app\n for contact tracing of people who\'ve been diagnosed with COVID-19.',
    'q5.txt':'Tell us any of your thoughts about using\n a smart phone app for documenting your COVID-19 status.',
    'q6.txt':'Please complete the following statement.\n "I would be okay providing personal information to a COVID19 related App if ..."',
    'q7.txt':'Please describe the conditions under which\n you would use this tracing app.',
    'q8.txt':'Please describe the conditions under which\n you would use this status app.'
    }



    topic_weights = []

    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    arr = pd.DataFrame(topic_weights).fillna(0).values

    arr = arr[np.amax(arr, axis=1) > 0.35]

    topic_num = np.argmax(arr, axis=1)
    tsne_model = TSNE(n_components=2, verbose=0, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    n_topics = num_topics
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

    x = tsne_lda[:,0]
    y=tsne_lda[:,1]

    fig, ax = plt.subplots()

    for topic_index in range(n_topics):
        filtered_x = [ x[xi] for xi in range(len(x)) if int(topic_num[xi])==int(topic_index)]
        filtered_y = [ y[yi] for yi in range(len(y)) if int(topic_num[yi])==int(topic_index)]
        topic_label = topic_dict[filename][topic_index]
        color = mycolors[topic_index]
        ax.scatter(filtered_x, filtered_y, c = color, label = topic_label)

    ax.legend()
    plt.legend(loc=0)
    plt.title(question_dict[filename])
    plt.show()



file_list = argv[1:]
all_lines = []
all_response_tokens = []

#Create output directory if it doesn't exist
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

for file in file_list:
    print('____________________________________{}____________________________________'.format(file))
    if use_range_of_topics:
        num_topics_iter = num_topics_range
    else:
        num_topics_iter = [num_topics]

    if create_visualization:
        num_topics_iter = [file_topics_dict[file]]

    for num_topics in num_topics_iter:
        lines, response_token_list = get_lines(file)
        processed_output = build_models(response_token_list)
        dominant_topic_df = create_topic_model(processed_output,num_topics,create_visualization, topic_dict, file)
        print(dominant_topic_df['Keywords'].unique())
        if write_csv:
            dominant_topic_df.to_csv(output_directory + '/' + 'q1_{}.csv'.format(num_topics))
