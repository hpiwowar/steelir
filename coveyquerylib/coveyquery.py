#!/usr/bin/env python
# Called Covey Query because you build it by "beginning with the end in mind"
"""
@author: Heather Piwowar
@contact:  hpiwowar@gmail.com
"""
from nltk.probability import ConditionalFreqDist, FreqDist
import string
import re
import operator
from cStringIO import StringIO
import ConfigParser
from BeautifulSoup import BeautifulStoneSoup
import utils
from utils.mylog import mylog
from utils import mydb
from utils.cache import TimedCache
import stopwords
from stopwords import stopwords_pubmed, stopwords_journals, stopwords_bioinfdata
import datasources
from datasources import pubmed


# Read in configuration parameters
CONFIG_FILE_NAME = "/Users/hpiwowar/Documents/Code/hpiwowar/SteelIR/trunk/src/coveyquerylib/coveyquery.cfg"
config = ConfigParser.ConfigParser()
config.read(CONFIG_FILE_NAME)

def get_stopwords():
    stopwords_source = {}
    stopwords_source["pubmed"] = stopwords_pubmed.stopwords
    stopwords_source["journals"] = stopwords_journals.stopwords
    stopwords_source["bioinfdata"] = stopwords_bioinfdata.stopwords

    stopwords = {}
    stopwords["unigrams"] = []
    stopwords["bigrams"] = []
    stopwords["pattern"] = " "
    for source in stopwords_source:
        stopwords["unigrams"] += stopwords_source[source]["unigrams"]
        stopwords["bigrams"] += stopwords_source[source]["bigrams"]
        if stopwords_source[source]["pattern"]:
            stopwords["pattern"] += "|" + stopwords_source[source]["pattern"]
    stopwords["pattern_compiled"] = re.compile(stopwords["pattern"])
    return stopwords

stopwords = get_stopwords()

class CoveyQuery(object):

    def __init__(self, articledir="./articles", skipdb=False):
        if not skipdb:
            try:
                self.mydb = mydb.MyDb(config.get("db", "db"))
                self.mydb.get_cursor().execute('''create table bagofwords (pmcid text, words text)''')
                self.mydb.get_cursor().execute('''create unique index pmcid_bagofwords_idx on bagofwords (pmcid)''')           
            except:
                mylog.info("Not able to create database bagofwords table or index, maybe because it already exists.")
            self.mydb.free()    
        
        self.articledir = articledir

    def write_words_to_db(self, pmcid, words):
        try:
            self.mydb.get_cursor().execute("insert or replace into bagofwords (pmcid, words) values (?, ?)", 
                              (pmcid, words))
            self.mydb.free()
        except Exception, e:
            mylog.error(e)
            
    def get_bag_of_words(self, rawtext):
        try:
            soup = BeautifulStoneSoup(rawtext).findAll(text=True)
        except UnicodeEncodeError, e:
            mylog.info("Failed parsing the rawtext file")
            mylog.info("Not including it in arff file")
            return None
        
        doctext = " ".join(soup)
        # Split on all punctuation except hyphens
        # Translate table is complicated because string is unicode
        stopchars = string.punctuation.replace("-", "")        
        translate_table = dict(zip(map(ord, stopchars), u" " * len(stopchars)))
        doctext = doctext.translate(translate_table)

        (filtered_unigrams_set, filtered_bigrams_set) = get_filtered_unigrams_and_bigrams(doctext)

        doctext = " ".join(filtered_unigrams_set + filtered_bigrams_set)
        return doctext


 
    def all_doctext_words_from_db(self, pmcids):
        words = FreqDist()
        str_list_pmcids = "("
        for pmcid in pmcids:
            str_list_pmcids +=  pmcid + ","
        str_list_pmcids += "0)"
        try:
            self.mydb.get_cursor().execute("select words from bagofwords where pmcid in %s" % (str_list_pmcids))
            rows = self.mydb.get_cursor().fetchall()
            for row in rows:
                for word in row[0].split():
                    words.inc(word)
        except Exception, e:
            mylog.error(e)
        self.mydb.free()
        return words

    def doctext_from_db(self, pmcid):
        response = ""
        try:
            self.mydb.get_cursor().execute("select words from bagofwords where pmcid=%s" % (pmcid))
            row = self.mydb.get_cursor().fetchone()
            if row:
                response = row[0]
            self.mydb.free()
        except Exception, e:
            mylog.error(e)
        return response
    
    def get_doctext(self, docid, local_location):
        doctext = self.doctext_from_db(docid)
        if doctext:
            is_processed_for_bigrams = "_" in doctext
            if is_processed_for_bigrams:
                mylog.info("Bag of words and bigrams already in db")
                return(doctext)
        if not local_location:
            mylog.info("No db or local file for %s, so skipping" % (docid))
            return("")
        mylog.info("COMPUTING bag of words and bigrams and writing to db")
        rawtext = open(self.articledir + "/" + local_location).read()
        doctext = self.get_bag_of_words(rawtext)
        self.write_words_to_db(docid, doctext)
        return doctext
            
    def get_freq_dist_from_dataset_tuples(self, dataset_file_tuples):
        freq_dist = FreqDist()
        number_of_documents = len(dataset_file_tuples)
        docid_count = 0
        for (docid, local_location) in dataset_file_tuples:
            docid_count += 1
            mylog.info("Getting doctext: %-5s of %s, docid %-8s" %(docid_count, number_of_documents, docid))
            doctext = self.get_doctext(docid, local_location)
            if doctext:
                add_words_to_freq_dist(freq_dist, doctext)
        return(freq_dist)

    def get_recall_precision_inputs(self, dataset_file_locations):
        positive_fd = self.get_freq_dist_from_dataset_tuples(dataset_file_locations[1])
        negative_fd = self.get_freq_dist_from_dataset_tuples(dataset_file_locations[0])
        number_of_positive_texts = len(dataset_file_locations[1])
        return(positive_fd, negative_fd, number_of_positive_texts)
                        
    def get_recall_precision_tokens(self, recall, precision, feature_sources=["pmcfulltext"], info=None):
        (aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts) = get_aggregate_freq_dists(feature_sources, info, obj=self)
        if aggregate_pos_fd:
            (recall_tokens, valid_tokens, precision_tokens) = do_recall_precision_calcs(aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts, recall, precision)
        else:
            precision_tokens = None
        return(precision_tokens)


@TimedCache(timeout_in_seconds=60*60*24*7)        
def get_aggregate_freq_dists(feature_sources=["pmcfulltext"], info=None, obj=None):
    precision_tokens = None
    fd_list = []
    for source in feature_sources:
        #print source
        if ("pmcfulltext" in source):
            (positive_fd, negative_fd, number_of_positive_texts) = obj.get_recall_precision_inputs(info)
        elif ("mesh_basic" in source):
            (positive_fd, negative_fd, number_of_positive_texts) = inputs_from_mesh(info, getter=pubmed.mesh_basic, suffix="[mesh]")
        elif ("mesh_major" in source):
            (positive_fd, negative_fd, number_of_positive_texts) = inputs_from_mesh(info, getter=pubmed.mesh_major, suffix="[major]")
        elif ("mesh_qualifier" in source):
            (positive_fd, negative_fd, number_of_positive_texts) = inputs_from_mesh(info, getter=pubmed.mesh_qualifier, suffix="[sh]")
        elif ("article_title" in source):
            (positive_fd, negative_fd, number_of_positive_texts) = inputs_from_text(info, getter=pubmed.article_title, suffix="[ti]")
        elif ("abstract" in source):  # Note:  pure abstract filter is not available in PubMed
            (positive_fd, negative_fd, number_of_positive_texts) = inputs_from_text(info, getter=pubmed.abstract, suffix="[abstract]")
        elif ("tiabs" in source):
            (positive_fd, negative_fd, number_of_positive_texts) = inputs_from_text(info, getter=pubmed.title_and_abstract, suffix="[tiabs]")
        else:
            print "bad feature_source"
            raise(Exception)
            positive_fd = None
            negative_fd = None
        fd_list.append((positive_fd, negative_fd,))
    aggregate_pos_fd = FreqDist()    
    aggregate_neg_fd = FreqDist()    
    for (pos_fd, neg_fd) in fd_list:
        for key in pos_fd:
            aggregate_pos_fd.inc(key, count=pos_fd[key])
        for key in neg_fd:
            aggregate_neg_fd.inc(key, count=neg_fd[key])
    return(aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts)

@TimedCache(timeout_in_seconds=60*60*24*7)        
def get_mesh_frequency_distributions(pmids, getter=pubmed.mesh_basic, suffix=""):
    mesh_list = getter(pmids)
    #print mesh_list
    freq_dist = FreqDist()
    for mesh_concat in mesh_list:
        if mesh_concat:
            for mesh_term in mesh_concat.split(";"):
                freq_dist.inc(mesh_term+suffix)
    return(freq_dist)

@TimedCache(timeout_in_seconds=60*60*24*7)        
def get_text_frequency_distributions(pmids, getter=pubmed.article_title, suffix=""):
    text_list = getter(pmids)
    freq_dist = FreqDist()
    for text in text_list:
        (filtered_unigrams_set, filtered_bigrams_set) = get_filtered_unigrams_and_bigrams(text)
        token_string = " ".join(filtered_unigrams_set + filtered_bigrams_set)
        if token_string:
            for token in token_string.split():
                freq_dist.inc(token+suffix)
    return(freq_dist)

def print_frequency_proportion(pmids, freq_dist, max_num):
    num_pmids = len(pmids)
    print "\n"
    for key in freq_dist.keys()[0:max_num]:
        print "%25s %2.0f%%" %(key, 100.0*freq_dist[key]/num_pmids) 

def inputs_from_text(positives_and_negative_pmids, getter=pubmed.article_title, suffix=""):
    (positive_pmid_list, negative_pmid_list) = positives_and_negative_pmids;
    positive_fd = get_text_frequency_distributions(positive_pmid_list, getter, suffix)
    negative_fd = get_text_frequency_distributions(negative_pmid_list, getter, suffix)
    number_of_positive_texts = len(positive_pmid_list)
    return(positive_fd, negative_fd, number_of_positive_texts)
        
def inputs_from_mesh(positives_and_negative_pmids, getter=pubmed.mesh_basic, suffix=""):
    (positive_pmid_list, negative_pmid_list) = positives_and_negative_pmids;
    positive_fd = get_mesh_frequency_distributions(positive_pmid_list, getter, suffix)
    negative_fd = get_mesh_frequency_distributions(negative_pmid_list, getter, suffix)
    number_of_positive_texts = len(positive_pmid_list)
    return(positive_fd, negative_fd, number_of_positive_texts)
    
def do_recall_precision_calcs(positive_fd, negative_fd, number_of_positive_texts, recall, precision):
    recall_tokens = tokens_with_min_recall(positive_fd, number_of_positive_texts, recall) 
    valid_tokens = [token for token in recall_tokens if not in_stopwords(token)]
    precision_tokens = tokens_with_min_precision(positive_fd, negative_fd, precision, valid_tokens)
    return(recall_tokens, valid_tokens, precision_tokens)
                           
def add_words_to_freq_dist(my_freq_dist, token_string):
    tokens = token_string.split()
    for token in tokens:
        my_freq_dist.inc(token)
    return my_freq_dist
    
def tokens_with_min_recall(positive_freq_dist, number_of_positive_texts, recall):
    min_hits_for_recall = number_of_positive_texts * recall
    tokens_with_enough_hits = [token for token in positive_freq_dist if (positive_freq_dist[token] > min_hits_for_recall)]
    return tokens_with_enough_hits

def get_precision_recall_tuples(positive_freq_dist, negative_freq_dist, number_of_positive_texts, min_recall=0.05, min_precision=0.05):
    tuples = []
    for token in positive_freq_dist:
        recall = float(positive_freq_dist[token])/number_of_positive_texts
        precision = float(positive_freq_dist[token])/(positive_freq_dist[token] + negative_freq_dist[token])
        if (recall >= min_recall) and (precision >= min_precision):
            fmeasure = 2*precision*recall/(precision + recall)
            tuples += [(token, precision, recall, fmeasure)]
    return tuples

def sort_by_fmeasure(tuples):
    sorted_tuples = sorted(tuples, key=operator.itemgetter(3), reverse=True)
    return sorted_tuples
    
def print_tuples(tuples, max=None, sort=None):
    if sort=="F":
        tuples.sort(key=operator.itemgetter(3), reverse=True)
    elif sort=="Pr":
        tuples.sort(key=operator.itemgetter(1), reverse=True)
    elif sort=="Re":
        tuples.sort(key=operator.itemgetter(2), reverse=True)
        
    if max:
        tuples = tuples[0:max]
        
    print "%50s\t%4s\t%4s\t%4s" %(" ", "Pr", "Re", "F")
    for (token, a, b, c) in tuples:
        print "%50s\t%.2f\t%.2f\t%.2f" %(token, a, b, c)

def write_tuples(tuples, filename="./results/tuples.txt"):
    fh = open(filename, "w")
    fh.write("%s|%s|%s|%s|%s\n" %("feature", "precision", "recall", "fmeasure", "stop"))
    for (token, precision, recall, fmeasure) in tuples:
        is_stopword = in_stopwords(token)
        fh.write("%s|%.4f|%.4f|%.4f|%s\n" %(token.replace("_", " "), precision, recall, fmeasure, str(is_stopword)))
    fh.close()
        
def tokens_with_min_precision(positive_freq_dist, negative_freq_dist, recall, features=None):
    def has_enough_recall(token):
        positive_N = positive_freq_dist[token]
        total_N = positive_freq_dist[token] + negative_freq_dist[token]
        has_enough = (positive_N >= total_N * recall)
        return has_enough
        
    if not features:
        features = positive_freq_dist.keys()
    features_with_enough_recall = [feature for feature in features if has_enough_recall(feature)]
    return features_with_enough_recall
        
def get_all_docids(dataset_file_locations):
    docids = []
    for classification in dataset_file_locations:
        docids += [docid for (docid, local_location) in dataset_file_locations[classification]]
    return docids

def get_feature_list(fd_all_words, min_number_instances):
    feature_list_prelim = [feature for feature in fd_all_words.samples() if (fd_all_words[feature] >= min_number_instances)]
    mylog.info("Prelim feature list is " + str(len(feature_list_prelim)) + " features long.")       
    feature_list = [feature for feature in feature_list_prelim if not in_stopwords(feature)]
    return(feature_list)
    
def in_stopwords(feature):
    words = feature.split("_")
    for word in words:
        if word in stopwords["unigrams"]:
            return True
    if feature in stopwords["bigrams"]:
        return True
    if stopwords["pattern_compiled"].search(feature):
        return True
    return False

remove_punctuation_pattern = re.compile(r"[,():.?]")

def get_filtered_unigrams_and_bigrams(text):
    text_remove_punctuation = remove_punctuation_pattern.sub("", text)
    all_text_list = text_remove_punctuation.lower().split()
    
    unigrams_set = set(all_text_list)  
    filtered_unigrams_set = [word for word in unigrams_set if is_ok(word) and not in_stopwords(word)]              

    all_bigrams_list = [(first, second) for (first, second) in zip(all_text_list[0:-1], all_text_list[1:])]
    bigrams_set = set(all_bigrams_list)
    filtered_bigrams_set = [first+"_"+second for (first, second) in bigrams_set if (is_ok(first) and is_ok(second) and not in_stopwords(first+"_"+second))]              
    return(filtered_unigrams_set, filtered_bigrams_set)
    
def is_ok(word):
    def containsAny(str, set):
        """Check whether 'str' contains ANY of the chars in 'set'"""
        # from http://code.activestate.com/recipes/65441/   
        #"[c in str for c in set]" creates a list of true/false values, one for each char in the set. "1 in [c in str for c in set]" then checks if at least one "true" is in that list.
        return 1 in [c in str for c in set]

    is_ok = True
    if len(word) < 3:
        is_ok = False
    if len(word) >= 30:
        is_ok = False
    if not containsAny(word, string.ascii_letters):
        is_ok = False
    return(is_ok)

                 
class CoveyQueryError(Exception):
    """Base class for errors in the CoveyQuery module."""

def _test():
    import doctest
    doctest.testmod(verbose=True)

if __name__ == "__main__":
    _test()
