pypage
======
  PAGE: Pathway Analysis of Gene Expression

The PAGE algorithm is an `information theoretic <https://en.wikipedia.org/wiki/Information_theory>`_ approach to performing pathway enrichment/depletion on an expression profile. 
The intuition behind the algorithm is that by considering the depletions and enrichments of a specific profile as being `mutually informative <https://en.wikipedia.org/wiki/Mutual_information>`_ to a pathway the search space of possibly enriched pathways can be reduced considerably.
The list of informative pathways can then be tested for with a `hypergeometric test <https://en.wikipedia.org/wiki/Hypergeometric_distribution>`_ to measure significant enrichment/depletion of genes in that pathway.
Using the mutual information approach to filter to a smaller set of possible pathways it is possible to tremendously reduce the impact of the `multiple comparisons problem <https://en.wikipedia.org/wiki/Multiple_comparisons_problem>`_ and as a result have a higher degree of certainty than `traditional GSEA methods <https://www.gsea-msigdb.org/gsea/index.jsp>`_. 

  The original PAGE algorithm was `published <https://doi.org/10.1016/j.molcel.2009.11.016>`_ in 2009. The original code can also be accessed on the `Tavazoie Lab Site <https://tavazoielab.c2b2.columbia.edu/iPAGE/>`_ and on the `Goodarzi Lab Github <https://github.com/goodarzilab/PAGE>`_


API Tree
========

.. toctree::
  :maxdepth: 3
  
  api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
