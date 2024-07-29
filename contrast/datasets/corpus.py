import pandas as pd

class Corpus:
    def __init__(self,
                 documents : dict = None,
                 queries : dict = None,
                 qrels : pd.DataFrame = None
                 ) -> None:
        self.documents = documents
        self.queries = queries
        self.qrels = qrels

        self.__post_init__()
    
    def __post_init__(self):
        if self.qrels:
            for column in 'query_id', 'doc_id', 'relevance':
                if column not in self.qrels.columns: raise ValueError(f"Format not recognised, Column '{column}' not found in qrels dataframe")
        
            self.qrels = self.qrels[['query_id', 'doc_id', 'relevance']]
    
    def has_documents(self):
        return self.documents is not None

    def has_queries(self):
        return self.queries is not None
    
    def has_qrels(self):
        return self.qrels is not None
    
    def queries_iter(self):
        for queryid, text in self.queries.items():
            yield {"query_id" : queryid, "text" : text}
    
    def docs_iter(self):
        for docid, text in self.documents.items():
            yield {"doc_id" : docid, "text" : text}
    
    def qrels_iter(self):
        for queryid, docid, relevance in self.qrels.itertuples(index=False):
            yield {"query_id" : queryid, "doc_id" : docid, "relevance" : relevance}
        
    