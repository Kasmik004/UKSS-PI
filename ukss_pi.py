import numpy as np
import nltk
import re
import networkx as nx
from typing import Dict, List, Union, Tuple
from collections import defaultdict
import math
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
stop_words = stopwords.words('english')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For additional word information
nltk.download('averaged_perceptron_tagger')  # For POS tagging
from nltk.corpus import wordnet
from nltk import pos_tag
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, List
import time


class UKSS_PI:
    def __init__(self,text):
        self.text = text
        self.words = self.__preprocess(text)
        
        
    def __get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun if unknown
        
    def __lemmatize_with_pos(self, text):
        lemmatizer = WordNetLemmatizer()
        pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        # Tokenize the text and get part of speech tags
        words = re.findall(r'\w+', text)
        word_pos = pos_tag(words)

        # Filter words based on POS tags and lemmatize only those
        lemmatized_words = [lemmatizer.lemmatize(word, self.__get_wordnet_pos(tag)) for word, tag in word_pos if tag in pos_tags]

        return ' '.join(lemmatized_words)
    
    
    def __preprocess(self,txt):
        txt = txt.lower()
        # Remove HTML tags
        txt = re.sub(r"<.*?>", " ", txt)
        # Remove special characters and digits
        txt = re.sub(r"[^a-zA-Z]", " ", txt)
        #lemmatize
        txt = self.__lemmatize_with_pos(txt)
        # tokenization
        txt = nltk.word_tokenize(txt)
        # Remove stopwords
        txt = [word for word in txt if word not in stop_words]
        # Remove words less than three letters
        txt = [word for word in txt if len(word) >= 3]

        return txt
    
    def __sentence_dictnator(self,sentence):
        dict_ = {}
        for word in sentence:
            if word in dict_:
                dict_[word] += 1
            else:
                dict_[word] = 1
        return dict_
    
    def __term_frequency(self):
        """Computes the term frequency of each word in a given string.

        Args:
            text: The input string.

        Returns:
            A dictionary mapping each word to its term frequency.
        """
        text = self.words
        words_count = len(text)
        text_dict = self.__sentence_dictnator(text)
        for word in text_dict:
            text_dict[word] = text_dict[word] / words_count

        #Commet if you want unnormalized data for term frequency
        max_freq = max(text_dict.values())
        for word in text_dict:
            text_dict[word] = text_dict[word] / max_freq
        # return {word: count/max_freq for word, count in freq.items()}

        return text_dict
    
    def __build_graph(self,words: List[str],window_size=3) -> nx.Graph:
        # Create graph
        graph = nx.Graph()

        # Add nodes (words)
        for word in words:
            if not graph.has_node(word):
                graph.add_node(word)

                # Add edges based on co-occurrence within window
        for i in range(len(words)):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                if graph.has_edge(words[i], words[j]):
                    graph[words[i]][words[j]]['weight'] += 1
                else:
                    graph.add_edge(words[i], words[j], weight=1)

        return graph
    
    
    def __extract_keywords(   self,
                        num_keywords: int = 5,
                        damping: float = 0.85,
                        max_iter: int = 100,
                        return_scores: bool = True) -> Union[Dict[str, float], List[str]]:
        """
        Extract keywords from text and return them with their scores.

        Args:
            num_keywords: Number of top keywords to return
            damping: Damping factor for PageRank
            max_iter: Maximum iterations for PageRank
            return_scores: If True, return dict of words and scores; if False, return list of words

        Returns:
            Either a dictionary of {word: score} or a list of words, depending on return_scores
        """
        # Preprocess text
        words = self.words

        # Build graph
        graph = self.__build_graph(words)

        # Apply PageRank
        scores = nx.pagerank(graph, alpha=damping, max_iter=max_iter)

        # Sort words by score
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if return_scores:
            # Return dictionary of top words with their scores
            return {word: round(score, 4)
                   for word, score in sorted_words[:num_keywords]}
        else:
            # Return just the list of top words
            return [word for word, _ in sorted_words[:num_keywords]]
        
        
    def __get_word_rankings(self) -> List[Tuple[str, float]]:
        """
        Get all words with their scores, sorted by importance.

        Args:
            text: Input text

        Returns:
            List of tuples (word, score) sorted by score in descending order
        """
        words = self.words
        graph = self.__build_graph(words)
        scores = nx.pagerank(graph)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def __combine_scores(self,tf_scores, graph_scores, alpha=0.5):
        combined_scores = {}

                # Get all unique words
        all_words = set(tf_scores.keys()) | set(graph_scores.keys())

        #print(all_words)

        for word in all_words:
            tf_score = tf_scores.get(word, 0)
            graph_score = graph_scores.get(word, 0)

            # Weighted geometric mean
            combined_scores[word] = (
            (tf_score ** (1 - alpha)) *
            (graph_score ** alpha)
            )

        return combined_scores
    
    def __keyword_extraction_t_g(self):
        tf_scores = self.__term_frequency()
        keywords_with_scores = self.__extract_keywords(
                num_keywords=10,
                return_scores=True
            )
        combined_scores = self.__combine_scores(tf_scores=tf_scores, graph_scores=keywords_with_scores)
        return combined_scores
    
    def __SegmentWordScore(self, num_segments=10):
        # # Split the text into words
        # txt = txt.lower()
        # # Remove HTML tags
        # txt = re.sub(r"<.*?>", " ", txt)
        # # Remove special characters and digits
        # txt = re.sub(r"[^a-zA-Z]", " ", txt)

        # txt = nltk.word_tokenize(txt)

        # # Remove stopwords
        # txt = [word for word in txt if word not in stop_words]
        # Remove words less than three letters
        # words = [word for word in txt if len(word) >= 3]

        words = self.words

        # Calculate segment size
        total_words = len(words)
        segment_size = math.ceil(total_words / num_segments)

        # Initialize frequency dictionary
        freq_dict = defaultdict(lambda: [0] * num_segments)

        # Count frequencies in each segment
        for i, word in enumerate(words):
            segment_idx = min(i // segment_size, num_segments - 1)
            freq_dict[word][segment_idx] += 1

        return dict(freq_dict)
    
    def __gries_dp_log(self,word_counts, precision=6, epsilon=1e-6):
        N = len(word_counts)
        total_occurrences = np.sum(word_counts)

        # If the word never appears, return DP = epsilon (small positive value)
        if total_occurrences == 0:
            return epsilon

        # Calculate the proportion of occurrences in each section
        proportions = [count / total_occurrences for count in word_counts]

        # Uniform distribution (ideal case)
        uniform_prob = 1 / N

        # Compute the sum of absolute differences from uniform distribution
        sum_diff = np.sum([abs(p_i - uniform_prob) for p_i in proportions])

        # Calculate Gries' DP
        DP = 1 - (sum_diff / (2 * (1 - uniform_prob)))

        # Handle small floating-point precision issues
        DP = max(DP, epsilon)  # Avoid DP becoming less than epsilon

        # Logarithmic scaling to avoid zeros when multiplying
        log_DP = math.log(DP + epsilon)  # Add epsilon to avoid log(0)

        # Round the DP to avoid floating-point precision issues
        log_DP = round(log_DP, precision)

        return log_DP
    
    def __GetDispersonScore(self,):
        segments = self.__SegmentWordScore()
        for key,values in segments.items():
            segments[key] = abs(self.__gries_dp_log(values))


        return segments
    
    def run_concurrent_calculations(
    text: Any,
    func1: Callable,
    func2: Callable,
    func3: Callable,
    ) -> list:
            """
            Runs three functions concurrently on the same text.
            
            Args:
                dataset: The data both functions will process
                func1: First calculation function
                func2: Second calculation function
                func3: Third calculation function
            
            Returns:
                A list of results from all three functions
            """
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit both functions to be executed
                future1 = executor.submit(func1, text)
                future2 = executor.submit(func2, text,num_keywords=10,return_scores=True)
                future3 = executor.submit(func3, text)
        
                # Wait for both functions to complete and get their results
                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()
            
            return [result1, result2, result3]
    
    def keyword_extraction_t_g_d(self,):
        # results = []
        '''
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit functions to be executed in parallel
            future1 = executor.submit(self.term_frequency, text)
            future2 = executor.submit(self.extract_keywords, text, num_keywords=10, return_scores=True)
            future3 = executor.submit(self.GetDispersonScore, text)

            # Wait for functions to complete and gather their results
            result1 = future1.result()
            result2 = future2.result()
            result3 = future3.result()

            results.append(result1)  # tf_scores
            results.append(result2)  # graph_scores
            results.append(result3)  # dispersion_score
        '''
        results = [self.__term_frequency(), self.__extract_keywords( num_keywords=10, return_scores=True), self.__GetDispersonScore()]

        # Use results to calculate combined keyword scores and adjust with dispersion score
        keywords = self.__combine_scores(tf_scores=results[0], graph_scores=results[1])
        dispersion_score = results[2]

        for word in keywords:
            if word in dispersion_score:
                keywords[word] = keywords[word] / dispersion_score[word]

        return keywords

        
    def get_keywords(self):
        keywords = self.keyword_extraction_t_g_d()
        for word, score in keywords.items():
            if score != 0:
                print(f"{word}: {score}")
                
                
   
   
text = '''
The gym is a dedicated space for individuals to enhance their physical fitness, mental well-being, and personal health. Gyms offer a variety of equipment and facilities designed to cater to a wide range of fitness goals, from weightlifting and strength training to cardiovascular and flexibility exercises. Common equipment includes free weights, resistance machines, treadmills, rowing machines, and stationary bikes. Many gyms also feature open areas for stretching, functional training, and group exercise classes like yoga, Pilates, Zumba, and spin, making it accessible for people of all fitness levels and preferences. Working out at the gym provides numerous benefits beyond the physical. Regular exercise helps reduce stress by releasing endorphins, the body's natural mood enhancers, which improve emotional health and mental clarity. A routine gym visit fosters a sense of discipline and consistency, helping individuals build better habits over time. Additionally, the gym environment can be highly motivating, as it brings together people with similar goals, creating a sense of community and encouragement. Trainers and fitness professionals at the gym are often available to provide guidance, correct form, and develop personalized workout plans, making the gym experience safer and more effective. For many, the gym becomes a haven, a place where they can focus on themselves, escape daily pressures, and achieve gradual, measurable progress in their fitness journeys.
'''             
kss = UKSS_PI(text)


kss.get_keywords()
