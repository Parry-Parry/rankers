import os

class Model:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs

    def load(self):
        if self.model_name not in LOAD_FUNCS:
            raise ValueError(f"Model {self.model_name} not supported. Supported models are {list(LOAD_FUNCS.keys())}")
        return LOAD_FUNCS[self.model_name](**self.kwargs)

class ModelHandler(object):
    def __init__(self) -> None:
        pass
        

def load_monot5(checkpoint : str ='castorini/monot5-base-msmarco', batch_size : int = 64, **kwargs):
    from pyterrier_t5 import MonoT5ReRanker 

    return MonoT5ReRanker(model=checkpoint, batch_size=batch_size)

def load_bi_encoder(checkpoint : str ='sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size : int = 64, **kwargs):
    from transformers import AutoModel, AutoTokenizer
    from pyterrier_dr import HgfBiEncoder, BiScorer

    model = AutoModel.from_pretrained(checkpoint).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device)
    return BiScorer(backbone, batch_size=batch_size)

def load_dense_retrieval(index_path : str, checkpoint : str ='sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size : int = 64, **kwargs):
    from pyterrier_dr import NumpyIndex, HgfBiEncoder

    backbone = HgfBiEncoder.from_pretrained(checkpoint, batch_size=batch_size)
    index = NumpyIndex(index_path)
    return backbone >> index

def load_electra(checkpoint : str ='crystina-z/monoELECTRA_LCE_nneg31', batch_size : int = 64, **kwargs):
    from pyterrier_dr import ElectraScorer

    return ElectraScorer(model_name=checkpoint, batch_size=batch_size)

def load_splade(checkpoint : str = 'naver/splade-cocondenser-ensembledistil', batch_size : int = 128, index : str = 'msmarco_passage', **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    from pyt_splade import SpladeFactory
    from pyterrier_pisa import PisaIndex

    index = PisaIndex(index, threads=4).quantized()
    splade = SpladeFactory(checkpoint)
    return splade.query_encoder(batch_size=batch_size) >> index

def load_bm25(index : str = 'msmarco_passage', **kwargs):
    import pyterrier as pt 
    if not pt.started(): pt.init()
    import multiprocessing as mp

    file_load = os.path.exists(index)

    from pyterrier_pisa import PisaIndex
    threads = kwargs.pop('threads', mp.cpu_count())
    if file_load:
        return PisaIndex(index, threads=threads).bm25(**kwargs)
    else:
        return PisaIndex.from_dataset(index, threads=threads).bm25(**kwargs)

LOAD_FUNCS = {
    'monot5': load_monot5,
    'bi_encoder': load_bi_encoder,
    'electra': load_electra,
    'splade': load_splade,
    'bm25': load_bm25,
    'dr': load_dense_retrieval,
}