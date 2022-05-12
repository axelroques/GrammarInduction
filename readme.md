# Grammar Induction

Python implementation of a time series motif discovery method using grammar induction. The motifs discovered have variable lengths: inter-motif subsequences have variable lengths, but the intra-motif subsequences are also not restricted to have identical length.

Inspired from _Li et. al. (2013)_. The algorithm has four main steps:

- Transform each time series into a symbolic representation using the SAX algorithm (_Lin et. al., 2003_). Rather than discretizing the whole time series, subsequences of length _n_ are extracted from the time series, normalized and converted into a SAX word, with a stride of _k_. Each subsequence is thus discretized individually using SAX, and all of these SAX words are concatenated to form one single SAX phrase.
- From this SAX phrase, numerosity reduction is employed: if a word occurs multiple times consecutively, we only keep its first occurrence. _E.g._, the phrase 'ABC ABC ABC ABB ABB ACB ABB' becomes 'ABC ABB ACB ABB'. Numerosity reduction is the key that makes variable-length motif discovery possible.
- Using the 'reduced' phrase, a grammar induction algorithm is used. Contrary to _Li et. al. (2013)_, we used the _Re-Pair_ algorithm from _Larsson & Moffat (1999)_. Re-Pair is an offline algorithm and as such could lead to better grammar rules compared to _Sequitur_ because all of the data is directly available. However, we note that if we were interested in streaming data _Sequitur_ would be a better choice. In this work, we sorted the rules in descending order of length and then in increasing order of occurrence - _i.e._ the most interesting rules are those that are the longest and that appear more frequently.
- Finally, in order to visualize the grammar rules - _i.e._ the motifs found - we need to reverse the rules on the reduced SAX representation of the time series back to the original, real-valued time series.

---

## Requirements

**Mandatory**:

- numpy
- Re-Pair (https://github.com/axelroques/Re-Pair)
- SAX (https://github.com/axelroques/SAX)

**Optional**:

- matplotlib

---

## Examples

**Processing the data**:

```python
df = pd.read_csv('your_data.csv')
gi = GrammarInduction(df, w=3, a=4, n=100, k=10,
                      alphabet_type='letters')
gi.process()
```

The `gi.sax` object contains results from the different steps of the SAX algorithm and the various SAX parameters:

- `gi.sax.df_INT` returns the dataframe after the normalization step.
- `gi.sax.df_PAA` returns the dataframe after the PAA step.
- `gi.sax.df_SAX` returns the dataframe after the discretization step.
- `gi.sax.w` returns the number of segments in the PAA - and SAX - representation (after the dimensionality reduction).
- `gi.sax.a` returns the number of symbols in the alphabet.
- `gi.sax.alphabet` returns the different symbols in the alphabet (determined by parameter _a_).
- `gi.sax.breakpoints` returns the values of the different breakpoints computed to discretize the time series.

The `gi.grammars` and `gi.ranked_grammars` objects contains the rules found for the different columns of the input dataframe.

**Visualizing the results**:

- Grammar rules

  ```python
  gi.show_rules(i_col)
  ```

- Motifs
  ```python
  gi.show_motifs(i_col, i_rule)
  ```

where _i_col_ is the index of the column in the dataframe (index starting after the mandatory _'t'_ column) and _i_rule_ the index of the rule in the corresponding `gi.ranked_grammars` object.
