
from __future__ import with_statement
import sys
import os
import nose
from nose.tools import assert_equals
from tests import slow, online, notimplemented, acceptance
from nltk.probability import FreqDist
import datasources
from datasources import pubmed
import coveyquerylib
from coveyquerylib import coveyquery


if __name__ == '__main__':
    nose.runmodule()

dataset_file_locations = {}
dataset_file_locations[0] = [('2652833', u'PLoS Genet/5-3/pe1000433-2652833/pgen.1000433.nxml'),
 ('2706974', u'PLoS Genet/5-7/pe1000572-2706974/pgen.1000572.nxml'),
 ('2483987', u'BMC Genomics/9-_/p318-2483987/1471-2164-9-318.nxml'),
 ('2684660', u'Genome Med/1-4/p39-2684660/gm39.nxml'),
 ('2650282', u'PLoS Genet/5-3/pe1000421-2650282/pgen.1000421.nxml'),
 ('2649212', u'PLoS Comput Biol/5-3/pe1000317-2649212/pcbi.1000317.nxml'),
 ('2683938', u'PLoS Genet/5-6/pe1000506-2683938/pgen.1000506.nxml'),
 ('1805816', u'PLoS ONE/2-2/pe196-1805816/pone.0000196.nxml'),
 ('2636809', u'BMC Bioinformatics/9-_/p530-2636809/1471-2105-9-530.nxml'),
 ('2702100', u'PLoS Genet/5-7/pe1000564-2702100/pgen.1000564.nxml')]
dataset_file_locations[1] = [('1808491', u'PLoS Comput Biol/3-3/pe30-1808491/pcbi.0030030.nxml'),
 ('2395256', u'Genome Biol/9-1/pR17-2395256/gb-2008-9-1-r17.nxml'),
 ('2715224', u'Nucleic Acids Res/37-13/p4194-2715224/gkn1076.nxml'),
 ('2194786', u'BMC Genomics/8-_/p322-2194786/1471-2164-8-322.nxml'),
 ('1352360', u'BMC Genomics/6-_/p182-1352360/1471-2164-6-182.nxml'),
 ('2257940', u'BMC Genomics/9-_/p37-2257940/1471-2164-9-37.nxml'),
 ('2621211', u'BMC Genomics/9-_/p598-2621211/1471-2164-9-598.nxml'),
 ('1802568', u'Nucleic Acids Res/35-1/p193-1802568/gkl1059.nxml'),
 ('1849891', u'PLoS ONE/2-4/pe386-1849891/pone.0000386.nxml'),
 ('1188246', u'PLoS Med/2-8/pe199-1188246/pmed.0020199.nxml')]

querybuilder = coveyquery.CoveyQuery(skipdb=True)

gold_schiz_pmids = ["17554300","17571346","17701901","17463248","18372903","18369103","17704812","17767166","16803859","17662880"]
gold_adhd_pmids = ["10845062","11283309","10860804","11449264","10769304","11994752","11525331","11906227","12668349","10591283"]

class TestCoveyQuery(object):    
        # Should ideally have a test database, clean up after myself, etc.  oh well, no time to do it now.
    @slow
    def test_get_doctext(self):
        docid, local_location = dataset_file_locations[0][0]
        doctext = querybuilder.get_doctext(docid, local_location)
        assert_equals(len(doctext), 37589)
        assert_equals(doctext[0:50], u'x0002a four marchini assembled localized diallelic')

    def test_add_to_freq_dist(self):
        fd = FreqDist()
        fd = coveyquery.add_words_to_freq_dist(fd, "hi how are you doing_today")
        assert_equals(len(fd.samples()), 5)
        assert_equals(fd.keys(), ['doing_today', 'how', 'you', 'hi', 'are'])
    
    def test_tokens_with_min_recall(self):
        doctexts = ["hello world", "hello world how are you today", "it is a small world after_all"]
        number_of_texts = len(doctexts)
        fd = FreqDist()
        for doctext in doctexts:
            fd = coveyquery.add_words_to_freq_dist(fd, doctext)
            
        tokens_95 = coveyquery.tokens_with_min_recall(fd, number_of_texts, 0.95)
        assert_equals(tokens_95, ['world'])

        tokens_40 = coveyquery.tokens_with_min_recall(fd, number_of_texts, 0.40)
        assert_equals(tokens_40, ['world', 'hello'])
        
    def test_tokens_with_min_precision(self):
        positive_fd = coveyquery.add_words_to_freq_dist(FreqDist(), "hello world how are you today")
        negative_fd = coveyquery.add_words_to_freq_dist(FreqDist(), "it is a small world after_all")
        tokens = coveyquery.tokens_with_min_precision(positive_fd, negative_fd, 0.95)
        assert_equals(tokens, ['how', 'are', 'you', 'hello', 'today'])
        
    @slow
    def test_get_positive_freq_dist(self):    
        positive_freq_dist = querybuilder.get_freq_dist_from_dataset_tuples(dataset_file_locations[1])
        assert_equals(positive_freq_dist.N(), 35005) 
        
    @slow
    def test_get_high_recall_positives(self):
        positive_fd = querybuilder.get_freq_dist_from_dataset_tuples(dataset_file_locations[1])
        number_of_texts = len(dataset_file_locations[1])
        recall_tokens = coveyquery.tokens_with_min_recall(positive_fd, number_of_texts, 0.90)
        assert_equals(len(recall_tokens), 88)
        
        valid_tokens = [token for token in recall_tokens if not coveyquery.in_stopwords(token)]
        assert_equals(len(valid_tokens), 72)
        
    @slow
    def test_get_high_precision_positives_pmcfulltext(self):
        features = querybuilder.get_recall_precision_tokens(0.5, 0.9, ["pmcfulltext"], dataset_file_locations)
        assert_equals(features, [u'pcr', u'conserved', u'gel', u'altered', u'downstream', u'genome-wide_expression', u'responsive', u'responses'])

    def test_get_mesh_freq_dist_schiz(self):
        dist = coveyquery.get_mesh_frequency_distributions(gold_schiz_pmids)
        coveyquery.print_frequency_proportion(gold_schiz_pmids, dist, 25)
        assert_equals(dist.items()[0:25], [('Humans', 10), ('Genome, Human', 6), ('Polymorphism, Single Nucleotide', 4), ('Genetic Predisposition to Disease', 4), ('Animals', 3), ('Case-Control Studies', 3), ('Female', 2), ('Signal Transduction', 2), ('Male', 2), ('Schizophrenia', 2), ('Diabetes Mellitus, Type 2', 2), ('Brain', 2), ('Meta-Analysis as Topic', 2), ('Regulatory Sequences, Nucleic Acid', 1), ('Transcription Factors', 1), ('Histones', 1), ('Chromatin', 1), ('Pilot Projects', 1), ('Antipsychotic Agents', 1), ('Ligands', 1), ('Receptors, Vasopressin', 1), ('Receptors, Metabotropic Glutamate', 1), ('Mood Disorders', 1), ('Genetic Markers', 1), ('Insulin-Like Growth Factor Binding Proteins', 1)])

    def test_get_mesh_freq_dist_adhd(self):
        dist = coveyquery.get_mesh_frequency_distributions(gold_adhd_pmids)
        coveyquery.print_frequency_proportion(gold_adhd_pmids, dist, 25)
        assert_equals(dist.keys()[0:25], ['Humans', 'Magnetic Resonance Imaging', 'Brain', 'Female', 'Male', 'Adult', 'Animals', 'Reference Values', 'Cognition', 'Prefrontal Cortex', 'Image Processing, Computer-Assisted', 'Models, Neurological', 'Photic Stimulation', 'Cerebral Cortex', 'Neurons', 'Memory', 'Attention', 'Brain Mapping', 'Neural Pathways', 'Artifacts', 'Emotions', 'Pattern Recognition, Visual', 'Saccades', 'Double-Blind Method', 'Action Potentials'])


    def test_get_high_precision_positives_mesh(self):
        features = querybuilder.get_recall_precision_tokens(0.1, 0.6, ["mesh_basic"], (gold_schiz_pmids, gold_adhd_pmids))
        assert_equals(features, ['Schizophrenia[mesh]'])

    def test_get_high_precision_positives_mesh_reverse(self):
        features = querybuilder.get_recall_precision_tokens(0.1, 0.75, ["mesh_basic"], (gold_adhd_pmids, gold_schiz_pmids))
        assert_equals(features, ['Adult[mesh]', 'Attention[mesh]', 'Memory[mesh]', 'Cognition[mesh]'])

    def test_get_high_precision_positives_mesh_major(self):
        dist = coveyquery.get_mesh_frequency_distributions(gold_adhd_pmids, getter=pubmed.mesh_major)
        coveyquery.print_frequency_proportion(gold_schiz_pmids, dist, 25)

        features = querybuilder.get_recall_precision_tokens(0.05, 0.05, ["mesh_major"], (gold_adhd_pmids, gold_schiz_pmids))
        assert_equals(features, ['Phenotype[major]', 'Psychiatry[major]'])

    def test_get_high_precision_positives_mesh_qualifier(self):
        features = querybuilder.get_recall_precision_tokens(0.1, 0.6, ["mesh_qualifier"], (gold_schiz_pmids, gold_adhd_pmids))
        assert_equals(features, ['genetics[sh]', 'chemistry[sh]', 'metabolism[sh]'])

    def test_get_high_precision_positives_title(self):
        features = querybuilder.get_recall_precision_tokens(0.1, 0.6, ["article_title"], (gold_schiz_pmids, gold_adhd_pmids))
        assert_equals(features, ['association[ti]', 'genome-wide_association[ti]', 'genome-wide[ti]', 'functional[ti]', 'diabetes[ti]', 'variants[ti]', 'schizophrenia[ti]', 'association_study[ti]', 'multiple[ti]', 'susceptibility[ti]'])

    def test_get_high_precision_positives_abstract(self):
        dist = coveyquery.get_text_frequency_distributions(gold_adhd_pmids, getter=pubmed.abstract)
        coveyquery.print_frequency_proportion(gold_adhd_pmids, dist, 25)

        features = querybuilder.get_recall_precision_tokens(0.4, 0.7, ["abstract"], (gold_schiz_pmids, gold_adhd_pmids))
        assert_equals(features, ['results[abstract]', 'individuals[abstract]', 'new[abstract]', 'association[abstract]', 'evidence[abstract]'])

    def test_get_high_precision_positives_combo(self):
        features = querybuilder.get_recall_precision_tokens(0.1, 0.6, ["mesh_basic", "article_title"], (gold_adhd_pmids, gold_schiz_pmids))
        assert_equals(features, ['Brain[mesh]', 'Male[mesh]', 'Adult[mesh]', 'Female[mesh]', 'brain[ti]', 'fmri[ti]', 'Attention[mesh]', 'voxel-based[ti]', 'Memory[mesh]', 'Cognition[mesh]', 'Neurons[mesh]'])
        
        
        