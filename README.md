
<!-- README.md is generated from README.Rmd. Please edit that file -->

# NLPTiltDataProcessing

<!-- badges: start -->
<!-- badges: end -->

The goal of NLPTiltDataProcessing is to summarise the nlp based data
(pre)processing steps for products. to increase their correctness and
consistency The steps are:

-   detecting typos

-   delimiting

-   deduplication.

Per step ther is an NLP based process to identify likely candidates for
typos correction, delimiting or deduplication. The outcome is manually
checked before applying the changes to the tilt database.

In the data folder simulated input data are stored to illustrate the
behavior. Please note that several aspects in this code, as well as the
setting of parameters, are tailored to the tilt database which is not
included in this repo.

# 1 Typo Detection

``` r
library(reticulate)
```

``` python
import pandas as pd
data = pd.read_csv("data/typo_cor_input.csv")
```

``` python
import spacy

# loading a big language model that holds a large share of English words
nlp = spacy.load('en_core_web_lg')
english_words = list(nlp.vocab.strings)
from textblob import TextBlob

def might_have_typos(text):
  in_vocab = [word in english_words for word in TextBlob(text).words]
  # as soon as one word in the text is not in the language model assume typos
  return (False in in_vocab)

data['suggest_typo'] = data["products_and_services"].apply(might_have_typos)
```

``` python
suggest_typo = data[data["suggest_typo"] == True]
print("Detecting potential typo for", len(suggest_typo.index), "out of", len(data.index), "rows.")
#> Detecting potential typo for 2 out of 6 rows.
```

``` r
knitr::kable(py$data)
```

| products\_id | products\_and\_services           | suggest\_typo |
|-------------:|:----------------------------------|:--------------|
|            1 | fish                              | FALSE         |
|            2 | apples                            | FALSE         |
|            3 | well management                   | FALSE         |
|            4 | Hofladen für biologische Produkte | TRUE          |
|            5 | selllling fishnchips              | TRUE          |
|            6 | energy consultants                | FALSE         |

# 2 Delimiting

Unfortunately the use of delimiters in the data is very inconsistent.
E.g. an *and* or a *,* might delimit 2 products but this is not
necessarily the case.

In this step all rows that holds potential delimiters are identified, in
a subsequent manual step delimiting is done if needed.

``` r
library(dplyr, warn.conflicts = FALSE)
```

``` r
delimit_data = readr::read_csv("data/delimiting_input.csv")
#> Rows: 8 Columns: 2
#> ── Column specification ────────────────────────────────────────────────────────
#> Delimiter: ","
#> chr (1): products_and_services
#> dbl (1): products_id
#> 
#> ℹ Use `spec()` to retrieve the full column specification for this data.
#> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

These are those most likely delimiters. The Dutch word for *or* (*of*)
is not included as it would yield too many false positives due to the
English word *of*.

``` r
delimiters <- "[[:space:]]and[[:space:]]|[[:space:]]und[[:space:]]|[[:space:]]y[[:space:]]|[[:space:]]et[[:space:]]|[[:space:]]en[[:space:]]|[[:space:]]or[[:space:]]|[[:space:]]oder[[:space:]]|[[:space:]]o[[:space:]]|[[:space:]]ou[[:space:]]|,|-|\\||/|;|&|\\$"
```

``` r
data_checked_for_delimiters <- delimit_data %>% 
  mutate(suggest_delimiter = grepl(delimiters, products_and_services))

assume_delimiters <- data_checked_for_delimiters %>% 
  dplyr::filter(suggest_delimiter == TRUE)
print(paste("Detecting potential delimiter for", nrow(assume_delimiters), "out of", nrow(delimit_data), "rows."))
#> [1] "Detecting potential delimiter for 5 out of 8 rows."
```

``` r
knitr::kable(data_checked_for_delimiters)
```

| products\_id | products\_and\_services | suggest\_delimiter |
|-------------:|:------------------------|:-------------------|
|            1 | fish                    | FALSE              |
|            2 | apples                  | FALSE              |
|            3 | fish & chips            | TRUE               |
|            4 | apples & pears          | TRUE               |
|            5 | honey, organic          | TRUE               |
|            6 | apples, pears           | TRUE               |
|            7 | mint, fresh and organic | TRUE               |
|            8 | mint                    | FALSE              |

# 3 Deduplication

There are a lot fuzzy duplicated products in the data. In the following,
clusters of words are constructed that are near duplicates. It is then
manually checked if words can be considered as duplicates. Harmonizing
the products to canonical forms will increase consistency and reduce
manual effort.

``` python
import pandas as pd
data = pd.read_csv("data/deduplication_input.csv")
```

``` python
documents = data["products_and_services"].to_list()
```

## Preprocessing

Removal of stop words is done to reduce noise. However given that our
products are not longer free text but relatively short formulations
there are not too many stopwords to be expected, instead removal of
standard stopwords could even be harmful, e.g. removing *a* in Vitamin
*a*, i.e. it will increase the number of false positive duplicates.
However since there will be a manual check of suggested duplicates this
is acceptable.

``` python
import spacy
nlp = spacy.load('en_core_web_lg')

def lemmatize(txt):
    lemmatised_list = [token.lemma_.lower() for token in nlp(txt) if not (token.is_stop or token.is_punct)]
    return(lemmatised_list)
```

``` python
texts = [[text for text in lemmatize(doc)] for doc in documents]
```

## Calculating Cosine Similarities

Note that at the moment all products are compared to all products, no
blocking is used. This could be adapted if needed, e.g. by blocking by
sector. It would limit the number of needed comparisons (speed) however
we would overlook cases where e.g. products are identical between
sectors.

### With unigrams

At the beginning of research on deduplication, bigrams were used. Using
unigrams leads to a higher number of products considered potential
matches, meaning less false negatives but more false positives. As we
optimise for recall, unigrams are used.

``` python
from gensim import corpora
from gensim.similarities import MatrixSimilarity
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(docString) for docString in texts]
```

Calculating pairwise cosine similarities. Only returning the n\_best
matches per product. The best 20 matches are returned. This number was
chosen after empirically checking behavior on on the original data.

``` python
n_best = 20
index = MatrixSimilarity(corpus=corpus,
                   num_features=len(dictionary),
                   num_best = n_best)
```

#### Wrangling the results of cosine similarity calculation

``` python
doc_id = 0
similar_docs = {}

for similarities in index:
    similar_docs[doc_id] = list(enumerate(similarities))
    doc_id += 1
    
```

Creating a dataframe of results from the pairwise cosine similarity
calculations.

``` python
import pandas as pd
row_original_list = []
row_match_list = []
similarity_list = []

for doc_id, sim_doc_tuple_list in similar_docs.items():
  for sim_doc_tuple in sim_doc_tuple_list:
    
     row_match = sim_doc_tuple[1][0]
     similarity = sim_doc_tuple[1][1]
     
     row_original_list.append(doc_id)
     row_match_list.append(row_match)
     similarity_list.append(similarity)
     
```

``` python
df_temp = pd.DataFrame({"row_original": row_original_list, "row_match": row_match_list, "similarity": similarity_list})
df_temp  = df_temp[df_temp["row_original"] != df_temp["row_match"]]
```

Adding original products\_and\_services.

``` python
lookup = data[["products_and_services"]]
```

``` python
df_temp = df_temp.merge(lookup, how = "inner", left_on = "row_original", right_index = True)
df_temp = df_temp.merge(lookup, how = "inner", left_on = "row_match", right_index = True)
```

``` r
knitr::kable(py$df_temp)
```

|     | row\_original | row\_match | similarity | products\_and\_services\_x  | products\_and\_services\_y  |
|:----|--------------:|-----------:|-----------:|:----------------------------|:----------------------------|
| 1   |             0 |          1 |  1.0000000 | apples                      | apple                       |
| 2   |             1 |          0 |  1.0000000 | apple                       | apples                      |
| 6   |             3 |          4 |  0.7500000 | exporter red chili pepper   | exporters of red hot pepper |
| 8   |             4 |          3 |  0.7500000 | exporters of red hot pepper | exporter red chili pepper   |
| 10  |             5 |          6 |  0.7071068 | pasteboard                  | pasteboard working          |
| 12  |             6 |          5 |  0.7071068 | pasteboard working          | pasteboard                  |
| 14  |             7 |          9 |  0.7071068 | geographical maps           | maps                        |
| 17  |             8 |          9 |  0.7071068 | map of the world            | maps                        |
| 15  |             7 |          8 |  0.5000000 | geographical maps           | map of the world            |
| 21  |             9 |          8 |  0.7071068 | maps                        | map of the world            |
| 18  |             8 |          7 |  0.5000000 | map of the world            | geographical maps           |
| 20  |             9 |          7 |  0.7071068 | maps                        | geographical maps           |

In the following data pairs are considered duplicates if their
similarity exceeds a certain threshold.

#### Identifying clusters

The dataframe *df\_temp* now holds all pairwise comparison as described
above. It does not show with the toy data set, but using the tilt
products (&gt;30000) we need to identify clusters of duplicates that can
be processed by a human. All products in a cluster could then be
converted to a canonical form.

In order to identify clusters **connected components** approach is
used**.** In concrete, if A and B are considered a duplicate and B and C
are considered a duplicate it is assumed that A, B and C are a cluster
and have a canonical form. This is a rather liberal approach for turning
results of pairwise comparisons into clusters but since there is a
manual approval step anyway and we aim for reducing the number of false
negatives (i.e. duplicates/clusters are not identified as such) this
approach is reasonable.

Since the number of products in the original data is rather high we need
to make sure to arrive at overseeable cluster sizes. Empiric research on
the tilt data showed that clusters &gt; 10 usually do not map well to a
canonical form, thus we accept only clusters up to a size of 10. In
order to group all duplicates into such clusters the similarity
threshold for considering 2 words a duplicate is increased iteratively
and per similarity the clusters that are below the cutoff size of 10 are
kept.

``` python
import networkx as nx
```

``` python
cluster_threshold = 10

df_continue = df_temp
df_accepted = pd.DataFrame(columns = ["cluster_id", "id", "sim_threshold", "cluster_threshold"])

sim_threshold = 0.5
cluster_id = 0
```

``` python
while sim_threshold < 1.01 and len(df_continue.index) > 0:
  
  print("Starting new round with a similarity threshold above", sim_threshold, ".")
 
  df_continue_temp = df_continue[df_continue["similarity"] > sim_threshold]
  print("-- Inferring clusters from", len(df_continue_temp.index), "rows.")
  duplicate_tuples_list = list(zip(df_continue_temp.row_original, df_continue_temp.row_match))
  G = nx.Graph()
  G.add_edges_from(duplicate_tuples_list)
  print("-- Created connected components.")
  cluster_list = [connected_component for connected_component in nx.connected_components(G)]
  clusters_list = []
  ids_list = []
  
  for cluster in cluster_list:
    for ids in cluster:
      clusters_list.append(cluster_id)
      ids_list.append(ids)
    cluster_id += 1
  
  df_w_clusters = pd.DataFrame({'cluster_id': clusters_list, 'id': ids_list})
  df_w_clusters["sim_threshold"] = sim_threshold
  df_w_clusters["cluster_threshold"] = cluster_threshold
  
  print("-- Identified", len(set(df_w_clusters["cluster_id"])), "clusters in this iteration.")
   
  cluster_size = df_w_clusters.groupby(['cluster_id']).size().reset_index(name='counts')
  big_cluster_ids = cluster_size[cluster_size["counts"] > cluster_threshold]["cluster_id"]
  print("-- From these clusters", len(set(big_cluster_ids)), "have a size higher than the cluster threshold", cluster_threshold,".")
  
  df_accepted_temp = df_w_clusters[~df_w_clusters["cluster_id"].isin(big_cluster_ids)]
  print("-- Accepted", len(df_accepted_temp.index), "new clustered rows in", len(set(df_accepted_temp["cluster_id"])), "clusters in this iteration.")
  
  df_accepted = pd.concat([df_accepted, df_accepted_temp], verify_integrity = True, ignore_index = True)
  print("-- This makes a temporary total of", len(df_accepted.index), "rows in", len(set(df_accepted["cluster_id"])), "clusters in this iteration.")
  
  ids_continue = df_w_clusters[df_w_clusters["cluster_id"].isin(big_cluster_ids)]["id"]
  df_continue = df_continue_temp[df_continue_temp["row_original"].isin(ids_continue) | df_continue_temp["row_match"].isin(ids_continue)]
  print("-- Continuing with", len(df_continue.index), "rows for which cluster sizes where above cluster threshold", cluster_threshold, "when using a minimal similarity of", sim_threshold,".")
  
  cluster_id += 1
  print("-- Using cluster id", cluster_id, "in the next round.")
  
  sim_threshold += 0.05
  print("-- Similarity threshold increased to", sim_threshold, ".")
#> Starting new round with a similarity threshold above 0.5 .
#> -- Inferring clusters from 10 rows.
#> -- Created connected components.
#> -- Identified 4 clusters in this iteration.
#> -- From these clusters 0 have a size higher than the cluster threshold 10 .
#> -- Accepted 9 new clustered rows in 4 clusters in this iteration.
#> -- This makes a temporary total of 9 rows in 4 clusters in this iteration.
#> -- Continuing with 0 rows for which cluster sizes where above cluster threshold 10 when using a minimal similarity of 0.5 .
#> -- Using cluster id 5 in the next round.
#> -- Similarity threshold increased to 0.55 .
```

``` python
lookup = data[["products_and_services", "products_id"]]

df_accepted = df_accepted.merge(lookup, how = "inner", left_on = "id", right_index = True)

df_accepted_w_clusters = df_accepted[["products_id", "products_and_services", "cluster_id", "sim_threshold", "cluster_threshold"]]
```

``` python
df_accepted_w_clusters = df_accepted_w_clusters.sort_values(by = ["cluster_id", "products_and_services"])
```

Given the small amount of data we only need 1 iteration in this demo and
arrive at the following clusters.

``` r
knitr::kable(py$df_accepted_w_clusters)
```

|     | products\_id | products\_and\_services     | cluster\_id | sim\_threshold | cluster\_threshold |
|:----|-------------:|:----------------------------|:------------|---------------:|:-------------------|
| 1   |            2 | apple                       | 0           |            0.5 | 10                 |
| 0   |            1 | apples                      | 0           |            0.5 | 10                 |
| 2   |            4 | exporter red chili pepper   | 1           |            0.5 | 10                 |
| 3   |            5 | exporters of red hot pepper | 1           |            0.5 | 10                 |
| 4   |            6 | pasteboard                  | 2           |            0.5 | 10                 |
| 5   |            7 | pasteboard working          | 2           |            0.5 | 10                 |
| 8   |            8 | geographical maps           | 3           |            0.5 | 10                 |
| 6   |            9 | map of the world            | 3           |            0.5 | 10                 |
| 7   |           10 | maps                        | 3           |            0.5 | 10                 |
