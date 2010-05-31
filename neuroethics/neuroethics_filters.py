#!/usr/bin/env python
"""
@author: Heather Piwowar
@contact:  hpiwowar@gmail.com
"""
from nltk.probability import ConditionalFreqDist, FreqDist
import datasources
from datasources import pubmed
import coveyquerylib
from coveyquerylib import coveyquery

from data.healthy_2002to2008_not5or6 import healthy_2002to2008_not5or6
from data.healthy_1991to2008_5or6 import healthy_1991to2008_5or6
from data.healthy_1991to2001_1or2 import healthy_1991to2001_1or2
from data.healthy_1991to2001_3or4 import healthy_1991to2001_3or4
from data.healthy_1991to2001_5or6 import healthy_1991to2001_5or6
from data.healthy_1991to2001_7 import healthy_1991to2001_7

querybuilder = coveyquery.CoveyQuery(skipdb=True)


def calc_results(precision_list, recall_list, pmids_pos_neg, feature_sources):
    (aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts) = coveyquery.get_aggregate_freq_dists(feature_sources, pmids_pos_neg, obj=None)
    results = {}
    for precision in precision_list:
        results[precision] = {}
        for recall in recall_list:
            (recall_tokens, valid_tokens, precision_tokens) = coveyquery.do_recall_precision_calcs(aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts, recall, precision)
            print("\n** Precision: %.2f, Recall: %.2f, N_precision: %4d"  %(precision, recall, len(precision_tokens)))
            print " ".join(precision_tokens)
            results[precision][recall] = precision_tokens
    return(results)

def show_feature_performance(pmids_pos_neg, feature_sources, min_recall=0.1, min_precision=0.1, sort_by="Pr"):
    (aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts) = coveyquery.get_aggregate_freq_dists(feature_sources, pmids_pos_neg, obj=None)
    tuples = coveyquery.get_precision_recall_tuples(aggregate_pos_fd, aggregate_neg_fd, number_of_positive_texts, min_recall, min_precision)
    sorted_tuples = coveyquery.sort_by_fmeasure(tuples)
    coveyquery.print_tuples(sorted_tuples, max=None, sort=sort_by)
    return(sorted_tuples)


feature_sources = ["mesh_basic", "mesh_major", "mesh_qualifier", "article_title", "tiabs", "abstract"]
#A = healthy_1991to2008_5or6
#B = healthy_2002to2008_not5or6
#A = healthy_1991to2001_3or4
#B = healthy_1991to2001_1or2 + healthy_1991to2001_5or6 # + healthy_1991to2001_7

#A = healthy_1991to2001_5or6
#B = healthy_1991to2001_1or2 + healthy_1991to2001_3or4


from data import neuroethicslike
#A = neuroethicslike.pubmed_neurethicslike_query_results
#A = neuroethicslike.pubmed_fmri_neuroethicslike_query_results

base_query = """("fmri"[text] OR "magnetic resonance imaging"[mesh]) AND ((neurosciences[mesh] OR neuroscience[Title/Abstract] OR neurology[mesh]) AND (ethics[sh] OR ethical[Title/Abstract] OR "bioethical issues"[mesh] OR "ethics, medical"[mesh] OR "legislation and jurisprudence"[Subheading])) OR neuroethic*[Title/Abstract]"""
base = pubmed.search(base_query)
A = pubmed.filter_pmids(base, "Personal Autonomy")

dist = coveyquery.get_mesh_frequency_distributions(A, getter=pubmed.mesh_basic)
print "\n\nTop list for (A, getter=pubmed.mesh_basic):"
coveyquery.print_frequency_proportion(A, dist, 150)

if False:
    dist = coveyquery.get_text_frequency_distributions(A, getter=pubmed.article_title)
    print "\n\nTop list for (A, getter=pubmed.article_title):"
    coveyquery.print_frequency_proportion(A, dist, 30)

    dist = coveyquery.get_text_frequency_distributions(A, getter=pubmed.abstract)
    print "\n\nTop list for (A, getter=pubmed.abstract):"
    coveyquery.print_frequency_proportion(A, dist, 30)

dist = coveyquery.get_text_frequency_distributions(A, getter=pubmed.title_and_abstract)
print "\n\nTop list for (A, getter=pubmed.title_and_abstract):"
coveyquery.print_frequency_proportion(A, dist, 150)



#from data import ethics_classifications

#A = ethics_classifications.not_ethics_pmids + ethics_classifications.maybe_ethics_pmids
#B = ethics_classifications.yes_ethics_pmids


#tuples = show_feature_performance((A, B), feature_sources, min_recall=0.15, min_precision=0.6, sort_by="Pr")
#print "\n\n"
#tuples = show_feature_performance((B, A), feature_sources, min_recall=0.15, min_precision=0.5, sort_by="Re")

#dist = coveyquery.get_mesh_frequency_distributions(A, getter=pubmed.mesh_basic)
#print "\n\nTop Ten for (A, getter=pubmed.mesh_basic):"
#coveyquery.print_frequency_proportion(A, dist, 10)


def calc_query_performance(gold_pmids, query, recall_filter_pmids=None):
    print query
    if not recall_filter_pmids:
        recall_filter_pmids = gold_pmids
    query_finds_recall_filter_pmids = pubmed.filter_pmids(recall_filter_pmids, query)
    query_finds_recall_filter_pmids_set = set(query_finds_recall_filter_pmids)
    gold_set = set(gold_pmids)
    print len(gold_pmids)
    print len(recall_filter_pmids)
    print len(query_finds_recall_filter_pmids)
    print len(gold_set & query_finds_recall_filter_pmids_set)
    print len(query_finds_recall_filter_pmids_set)
    precision = float(len(gold_set & query_finds_recall_filter_pmids_set)) / len(query_finds_recall_filter_pmids_set)
    
    recall = float(len(gold_set & query_finds_recall_filter_pmids_set)) / len(gold_pmids)
    return(precision, recall)

def base():
    base_query = """(("humans"[mesh] AND "magnetic resonance imaging"[mesh] AND Journal Article[ptyp] NOT "mental disorders"[mesh]) 
    NOT (Editorial[ptyp] OR Letter[ptyp] OR Meta-Analysis[ptyp] OR Practice Guideline[ptyp] OR Review[ptyp] OR Case Reports[ptyp] OR Comment[ptyp] OR Corrected and Republished Article[ptyp])
    AND ("1991"[PDAT] : "2001"[PDAT])
    AND English[lang])"""

    query_for_1or2s = """(Movements OR Finger OR (Somatosensory Cortex[mesh]) OR somatosensory[Title/Abstract] OR (Cerebrovascular Circulation[mesh]) OR "primary motor"[Title/Abstract] OR "primary visual"[Title/Abstract] OR sensorimotor[Title/Abstract] OR "motor area"[Title/Abstract] OR oxygenation[Title/Abstract] OR (Motor Cortex[mesh]) OR "visual cortex"[Title/Abstract] OR (Acoustic Stimulation[mesh]))"""
    query_for_3or4s = """((Temporal Lobe[mesh]) OR (Prefrontal Cortex[mesh]) OR (Pattern Recognition, Visual[mesh]) OR (Visual Perception[mesh]) OR semantic[Title/Abstract] OR verbal[Title/Abstract] OR "left inferior"[Title/Abstract] OR retrieval[Title/Abstract] OR memory[Title/Abstract] OR language[Title/Abstract] OR Memory[mesh])"""
    #query_for_5or6s = """(Amygdala OR Emotional OR emotionally[Title/Abstract] OR (Facial Expression[mesh]) OR faces[Title/Abstract] OR facial[Title/Abstract] OR affective[Title/Abstract] OR Affect[mesh] OR dissociation[Title/Abstract] OR Arousal[mesh] OR (Decision Making[mesh]) OR happy[Title/Abstract] OR attentional[Title/Abstract] or networks[Title/Abstract] or Cognition[mesh])"""
    #query_for_5or6s = """(Amygdala OR Emotional OR emotionally[Title/Abstract] OR (Facial Expression[mesh]) OR faces[Title/Abstract] OR facial[Title/Abstract] OR affective[Title/Abstract] OR Affect[mesh] OR Arousal[mesh] OR (Decision Making[mesh]) OR happy[Title/Abstract] OR attentional[Title/Abstract] or Cognition[mesh])"""
    query_for_5or6s = """(Amygdala OR Emotions[mesh] OR emotion*[Title/Abstract] OR (Facial Expression[mesh]) OR faces[Title/Abstract] OR facial[Title/Abstract] OR Affect[mesh]"""
    #query_for_5or6s = """(Amygdala OR Emotional OR emotionally[Title/Abstract] OR (Facial Expression[mesh]) OR faces[Title/Abstract] OR facial[Title/Abstract] OR Affect[mesh] OR Arousal[mesh] OR (Decision Making[mesh])"""
    query_to_try = base_query + " AND " + query_for_5or6s# + " NOT (" + query_for_1or2s + " OR " + query_for_3or4s + ")"

    #(pr, re) = calc_query_performance(A, query_to_try, A+B)
    (pr, re) = calc_query_performance(A, query_for_5or6s, A+B)
    print "Precision:", round(pr, 2)
    print "Recall:", round(re, 2)



    #filtered = pubmed.filter_pmids(B, "Emotions[mesh]")
    filtered = pubmed.filter_pmids(B, query_for_5or6s)
    for pmid in filtered:
    #    print pmid
        pass

    if (False):
        pmids = pubmed.filter_pmids(B, "Visual Cortex[mesh]")
        for pmid in pmids:
            print pmid
        print "\n"
        for pmid in pmids:
            print pmid,
            if pmid in healthy_1991to2001_3or4: print "3or4",
            if pmid in healthy_1991to2001_5or6: print "5or6",
            print

    #precision_list = [0.6, 0.7, 0.8, 0.9, 0.95]     
    #recall_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   

    #print ("\n\n\n==============\n\nNow identify features for a range of precision and recall\n\n")
    #results = calc_results(precision_list, recall_list, (pmids_5or6, pmids_not6or6), feature_sources)        

    #print ("\n\n\n==============\n\nNow identify features to retrieve the opposite set = switch the order\n\n")
    #results2 = calc_results(precision_list, recall_list, (pmids_not6or6, pmids_5or6), feature_sources)        


    #print results[0.4]

    #for precision in results:
    #    for recall in results[precision]:
    #        print("\n** Precision: %.2f, Recall: %.2f, N_precision: %4d"  %(precision, recall, len(results[precision][recall])))
    #        print " ".join(results[precision][recall])
    #pos = healthy_1991to2008_5or6
    #neg = healthy_2002to2008_not5or6
    #feature_set = ["mesh_basic", "mesh_major", "mesh_qualifier", "article_title", "abstract"]
    #print "\n Computing features using ", feature_set
    #print "For Positive:healthy_2002to2008_5or6, Negative:healthy_1991to2008_not5or6"
    #print "Recall above 0.2 and Precision above 0.55"
    #features = querybuilder.get_recall_precision_tokens(0.2, 0.55, feature_set, (pos, neg))
    #print "\n" + " ".join(features)

    #dist = coveyquery.get_mesh_frequency_distributions(healthy_1991to2008_5or6, getter=pubmed.mesh_basic)
    #print "\n\nTop Ten for (healthy_1991to2008_5or6, getter=pubmed.mesh_basic):"
    #coveyquery.print_frequency_proportion(healthy_1991to2008_5or6, dist, 10)

