
from sax import SAX_subsequences
from .tree.tree import Tree
from repair import RePair

from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class GrammarInduction:

    def __init__(self, df, w=3, a=4, n=100, k=10):

        # Raw data
        self.df = df

        # SAX parameters
        self.sax = SAX_subsequences(df, w=w, a=a, n=n, k=k,
                                    alphabet_type='letters')

        # Some parameters that will be used when plotting
        self.trees = []
        self.sax_time_vectors = None
        self.ranked_rules = None

    def process(self):
        """
        Grammar Induction process.
        """

        # SAX subsequence discretization
        self.sax.process()

        # Generate SAX phrases
        self.SAX_phrases = self.generate_SAX_phrases(self.sax.df_SAX)

        # Numerosity reduction
        self.reduced, self.mapping = self.numerosity_reduction(
            self.SAX_phrases)

        # Find the grammar rules
        self.grammars = [RePair(self.reduced[key])
                         for key in self.reduced.keys()]

        # Rank the rules
        self.ranked_rules = self.rank_grammar_rules(self.grammars)

        return

    @staticmethod
    def generate_SAX_phrases(df_SAX):
        """
        Generate SAX phrases from the SAX words for 
        each column in the input dataframe.
        """

        SAX_phrases = {
            f'{col}': [''.join(c for c in subsequence.loc[:, col])
                       for subsequence in df_SAX]
            for col in df_SAX[0].columns[1:]
        }

        return SAX_phrases

    @staticmethod
    def numerosity_reduction(SAX_phrases):
        """
        Conducts the numerosity reduction step, i.e.,
        if a string occurs multiple times consecutively,
        records only its first occurrence.

        Also creates a hash table called mapping, that will 
        become useful to map the position of the SAX words
        from their 'reduced' representation back to their
        original representation. 
        """

        mapping = {
            f'{col}': defaultdict(dict) for col in SAX_phrases.keys()
        }
        reduced = {
            f'{col}': [] for col in SAX_phrases.keys()
        }

        # Iterate over the columns
        for key in SAX_phrases.keys():

            # Iterate over the SAX words
            i_reduced = 0
            i_start = 0
            last_word = None
            for i, word in enumerate(SAX_phrases[key]):

                # If we are at the first iteration
                if not last_word:

                    # Add word to the reduced list
                    reduced[key].append(word)

                    # Add starting position of the word
                    mapping[key][word][i_reduced] = [i]

                    # Update loop parameters
                    i_start = i
                    last_word = word
                    i_reduced += 1

                # Add word to the reduced list if it was
                # not added right before
                if word != reduced[key][-1]:
                    reduced[key].append(word)

                    # Add starting position of the word
                    mapping[key][word][i_reduced] = [i]

                    # Add ending position of the last_word
                    mapping[key][last_word][i_reduced-1].append(i-i_start)

                    # Update loop parameters
                    i_start = i
                    last_word = word
                    i_reduced += 1

            # Termination
            mapping[key][word][i_reduced-1].append(i-i_start+1)

        return reduced, mapping

    @staticmethod
    def rank_grammar_rules(grammars):
        """
        Rank the grammar rules by 'interestingness'. 
        Here, rules are interesting if they are long and
        occur frequently.

        Returns a list of dataframes with the sorted grammar rules.
        """

        # Iterate over the grammar rules for each column of
        # the input dataframe
        dataframes = []
        for grammar in grammars:

            # Arrays for the indirect stable sort
            sizes = np.array([len(r)
                              for r in grammar.phrase.results['Expanded Rule']])
            occurrences = np.array(grammar.phrase.results['Occurrence'])

            # Sorting indices: sort first by decreasing length,
            # then by occurrence.
            # The last rule corresponds to the initial phrase so
            # we do not report it
            i_order = np.lexsort((-occurrences, -sizes))[:-1]

            # Rank the rules
            dataframes.append(grammar.get_results().iloc[i_order, :-1])

        return dataframes

    def show_rules(self, i_col, ordered=True):
        """
        Returns a dataframe with the ranked rules for the 
        desired column. 
        """

        if isinstance(self.ranked_rules, list):
            if ordered:
                return self.ranked_rules[i_col]
            else:
                return self.grammars[i_col].get_results()
        else:
            raise RuntimeError('No rules found, run process method first.')

    def show_motifs(self, i_col, i_rule):
        """
        Once the grammar induction process is over,
        we might be interested in visualizing the different 
        grammar rules that were found.

        This function does just that. Plots a figure where the 
        motifs for the i_rule rule for the i_col column of the
        input dataframe are highlighted.
        """

        # Encode the reduced representation in suffix trees
        if not self.trees:
            self.trees = self.generate_suffix_trees(self.reduced)

        # Retrieve correct variables for the search
        tree = self.trees[i_col]
        motif = self.ranked_rules[i_col].iloc[i_rule, 1]
        mapping = self.mapping[self.df.columns[i_col+1]]

        print(f'Column {self.df.columns[i_col+1]}')
        print(f'Grammar rule = {motif}')

        # Search for the motif using the tree
        _, starting_positions = tree.find_motifs(motif)

        """
        We must correct the starting positions of the motif to 
        take into account the size of a symbol (+ 1 because of 
        the space separation between two successive symbols)
        
        Note: we have to do this because the tree encodes each 
        character as a unique symbol. But the way the tree is
        implemented could cause some other issues.
        E.g.: 'ABC ACB' is encoded as 'A', 'B', 'C', 'A','C''B'
        in the tree rather than 'ABC', 'ACB'. This may create
        erroneous matches when searching for patterns, for 
        instance: 'BCA' could be detected in the previous 
        string even though it should not be.
        """
        starting_positions = [pos // (len(motif.split(' ')[0])+1)
                              for pos in starting_positions]

        # Reverse the numerosity reduction
        motif_indices = self.find_true_motif_positions(mapping,
                                                       motif,
                                                       starting_positions)

        # Find the time values associated with these indices
        if not self.sax_time_vectors:
            self.sax_time_vectors = [df['t'].tolist()
                                     for df in self.sax.df_SAX]

        time_indices = self.find_time_values(self.sax_time_vectors,
                                             motif_indices)

        # Plot the motifs
        self.plot_motifs(self.sax.df, time_indices,
                         i_col, i_rule)

        return

    @staticmethod
    def generate_suffix_trees(reduced):
        """
        Encodes the reduced representation in suffix trees
        for easier access later on. Notably, this will be 
        useful when plotting the motifs associated with the
        grammar rules.
        """

        return [Tree({1: ' '.join(s for s in reduced[key])})
                for key in reduced.keys()]

    @staticmethod
    def find_true_motif_positions(mapping, motif, starting_positions):
        """
        Find the position of the motif in the original real-valued
        time series.

        We must take into account the numerosity reduction using the
        mapping dictionary. 

        Returns a list of tuples containing the start and end indices
        of the motif.
        """

        motif_indices = []

        # For every occurrence of the rule
        for pos in starting_positions:

            try:
                # Retrieve its starting position (in real coordinates)
                starting_pos = mapping[motif.split(' ')[0]][pos][0]

                # Get the length of the motif using the mapping hash table
                motif_length = 0
                for symbol in motif.split(' '):

                    symbol_pos = mapping[symbol]
                    motif_length += symbol_pos[pos][1]

                    pos += 1

                motif_indices.append(
                    tuple((starting_pos,
                           starting_pos + motif_length))
                )

            except KeyError:
                """
                This may occur when, by chance, a motif is found in the
                tree that does not correspond to a real motif in the 
                reduced representation. This is a direct consequence of 
                the way the tree is constructed (cf. comment in the main
                function).
                """
                continue

        return motif_indices

    @staticmethod
    def find_time_values(sax_time_vectors, motif_indices):
        """
        Returns the time values associated with the 
        motif indices.
        """

        time_indices = []

        # Iterate over the indices
        for indices in motif_indices:

            t_start = sax_time_vectors[indices[0]][0]
            t_end = sax_time_vectors[indices[1]-1][-1]

            # Add some time to the end because the SAX
            # algorithm only computes the starting time
            # of a segment
            t_end += sax_time_vectors[indices[0]][1] - \
                sax_time_vectors[indices[0]][0]

            time_indices.append(tuple((t_start, t_end)))

        return time_indices

    @staticmethod
    def plot_motifs(df, time_indices, i_col, i_rule):
        """
        Plots the motifs using the time indices.
        """

        # Get sub-dataframes using the time indices
        cut_dataframes = []
        for indices in time_indices:
            cut_dataframes.append(
                df.loc[(df['t'] >= indices[0]) &
                       (df['t'] < indices[1])]
            )

        # Plot
        _, ax = plt.subplots(figsize=(13, 8))

        ax.plot(df['t'], df.iloc[:, i_col+1], c='k', alpha=0.7)

        for d in cut_dataframes:
            ax.plot(d['t'], d.iloc[:, i_col+1], c='royalblue', alpha=0.8)

        ax.set_title(f'Column {d.columns[i_col+1]} - Rule {i_rule}')
        ax.set_xlabel('Time', fontsize=15)

        return
