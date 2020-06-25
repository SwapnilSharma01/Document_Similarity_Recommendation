# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:54:46 2020

@author: Swapnil

Recommendation Engine code
"""

"""
'pip install PyPDF2' will install the PyPDF2 package, this is one time activity.
"""
# pip install PyPDF2

import PyPDF2

#"decision_trees.pdf" #filename of your PDF/directory where your PDF is stored
#PDFfilename = "E:\\Swapnil\\Python\\decision_trees.pdf"
PDFfilename = "E:\Swapnil\Data_Science\Machine_Learning\Recommendation_Engines\Content_based\Policy_Recommendation\decision_trees.pdf" 


"""
# Code to read 1st page of pdf - start

#PdfFileReader object : This creates the file reader object and in open mode
#pfr = PyPDF2.PdfFileReader(open(PDFfilename, "rb"))

pdfFileObj = open(PDFfilename, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# create page object and extract text - for single page
pageObj = pdfReader.getPage(0)

# Create str object with data of 1st page of pdf
page1 = pageObj.extractText()

# Code to read 1st page of pdf - end
"""

# Code to read all pages of pdf - start

# import pandas and create empty dataframe
import pandas as pd
pdf_data = pd.DataFrame([])

# create list for intermediate usage
pdf_data_list = []

# Open pdf file using PdfFileReader and find the number of pages in the pdf
with open(PDFfilename,'rb') as pdf_file: 
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    for page_number in range(number_of_pages):  # using the total number of pages in the file, using page number : load the respective page's data into LIST
        page = read_pdf.getPage(page_number)
        page_content = page.extractText()
        pdf_data_list.append(page_content)
        #pdf_data = pd.DataFrame.append(page_content)
        #text_file.write(page_content)
        

# see all the content available in list        
#pdf_data_list      

# Creating a dataframe object from list
pdf_data_df = pd.DataFrame(pdf_data_list)
pdf_data_df.columns = ["page_content"]

# see all the content available in dataframe
#pdf_data_df



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  

# to find list of all the stopwords
# set(stopwords.words('english'))

# download the stopwords
# nltk.download('stopwords')

stop = stopwords.words('english')
#remove punctuation
pdf_data_df['page_content_processed'] = pdf_data_df.page_content.str.replace("[^\w\s]", "")

# change the page_content_processed in lower case
pdf_data_df['page_content_processed'] = pdf_data_df.page_content_processed.apply(lambda x: x.lower())

#Handle strange character in source
pdf_data_df['page_content_processed'] = pdf_data_df.page_content_processed.str.replace("‰Ûª", "''")

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
pdf_data_df['page_content_processed'] = pdf_data_df['page_content_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# word lemmatization on dataframe
import nltk
nltk.download('popular', quiet=True) 

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    # [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

#Sample 
"""
df1 = pd.DataFrame(['this was cheesy', 'she likes these books', 'wow this is great'], columns=['text'])
df1['text_lemmatized'] = df1.text.apply(lemmatize_text)
df1['text_lemmatized'].apply(lambda x: ' '.join(x))
"""

pdf_data_df['page_content_processed'] = pdf_data_df.page_content_processed.apply(lemmatize_text)


"""
After applying Lemmatization, we need to corect words like modeling, reporting etc.
Thus we will use Stemming for the same.

"""

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

porter_stemmer = PorterStemmer() 

# commenting the Stemming as it is stemming incorrectly at some places
#def stem_text(text):
#    return ' '.join([porter_stemmer.stem(w1) for w1 in text.split()])

#pdf_data_df['page_content_processed'] = pdf_data_df['page_content_processed'].apply(stem_text)


# same function differently
"""
def stem_text(text):
    tokens = text.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)
"""




#print("better :", lemmatizer.lemmatize("better", pos ="a")) 


#creating a temporary colun to combine multiple rows into single row in dataframe

pdf_data_df['group_key'] = 'temp_group'

# Creating a data frame with processed text of all pages of pdf
pdf_all_pages_comb_df = pd.DataFrame(columns=["file_name", "all_pages_comb_text"])

file_name = 'decision_trees.pdf'
comb_text_df = pdf_data_df.groupby('group_key')['page_content_processed'].apply(' '.join).reset_index()
comb_text_df["file_name"] = file_name
comb_text_df["all_pages_comb_text"] = comb_text_df["page_content_processed"]
comb_text_df = comb_text_df.drop(['page_content_processed'], axis = 1) 

# pdf_all_pages_comb_df will hold the processed text of all the pages of pdf
# any new data for new pdf will be appended to the pdf_all_pages_comb_df
pdf_all_pages_comb_df = pdf_all_pages_comb_df.append(pd.DataFrame(comb_text_df[['file_name', 'all_pages_comb_text']], columns=pdf_all_pages_comb_df.columns), ignore_index=True)




# Creating df with tf-idf values

from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVectorizer = TfidfVectorizer()
TfidfVectorizer_matrix = TfidfVectorizer.fit_transform(pdf_all_pages_comb_df['all_pages_comb_text'])

feature_names = TfidfVectorizer.get_feature_names()
index_name = pdf_all_pages_comb_df['file_name']
#index_name = index_name.tolist()

# Convert TfidfVectorizer_matrix  to Pandas Dataframe to see the word frequencies

doc_term_matrix = TfidfVectorizer_matrix.todense()
#doc_term_matrix = doc_term_matrix.tolist()

TfidfVectorizer_df = pd.DataFrame(doc_term_matrix, 
                  columns=feature_names,
                  index=index_name)


# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity


cosine_similarity_df_columns = TfidfVectorizer_df.index
cosine_similarity_df_index = TfidfVectorizer_df.index
#print(cosine_similarity(TfidfVectorizer_df, TfidfVectorizer_df))

cosine_similarity_df = pd.DataFrame(cosine_similarity(TfidfVectorizer_df, TfidfVectorizer_df), 
             columns=cosine_similarity_df_columns,
                  index=cosine_similarity_df_index)






