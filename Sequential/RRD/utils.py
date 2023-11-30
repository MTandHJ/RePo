

import torch

import freerec
from freerec.data.tags import USER, ITEM, ID, TIMESTAMP



def load_datapipes(cfg):

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    if cfg.model in ('MF',):
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomIDs(
            field=User, datasize=dataset.train().datasize
        ).sharding_filter().gen_train_uniform_sampling_(
            dataset, num_negatives=cfg.L
        ).batch(cfg.batch_size).column_().tensor_()

        validpipe = freerec.data.dataloader.load_gen_validpipe(
            dataset, batch_size=512, ranking=cfg.ranking
        )
        testpipe = freerec.data.dataloader.load_gen_testpipe(
            dataset, batch_size=512, ranking=cfg.ranking
        )
    elif cfg.model in ('GRU4Rec',):
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=dataset.train().to_roll_seqs(minlen=2)
        ).sharding_filter().seq_train_uniform_sampling_(
            dataset, leave_one_out=True, num_negatives=cfg.L # yielding (users, seqs, positives, negatives)
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen,
        ).add_(
            indices=[1, 2, 3], offset=cfg.NUM_PADS
        ).batch(cfg.batch_size).column_().rpad_col_(
            indices=[1], maxlen=None, padding_value=0
        ).tensor_()

        validpipe = freerec.data.dataloader.load_seq_rpad_validpipe(
            dataset, cfg.maxlen, 
            cfg.NUM_PADS, padding_value=0,
            batch_size=100, ranking=cfg.ranking
        )
        testpipe = freerec.data.dataloader.load_seq_rpad_testpipe(
            dataset, cfg.maxlen, 
            cfg.NUM_PADS, padding_value=0,
            batch_size=100, ranking=cfg.ranking
        )
    elif cfg.model in ('SASRec',):
        # trainpipe
        trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
            source=dataset.train().to_seqs(keepid=True)
        ).sharding_filter().seq_train_uniform_sampling_(
            dataset, leave_one_out=False, num_negatives=cfg.L # yielding (user, seqs, targets, negatives)
        ).lprune_(
            indices=[1, 2, 3], maxlen=cfg.maxlen
        ).add_(
            indices=[1, 2, 3], offset=cfg.NUM_PADS
        ).lpad_(
            indices=[1, 2, 3], maxlen=cfg.maxlen, padding_value=0
        ).batch(cfg.batch_size).column_().tensor_()

        validpipe = freerec.data.dataloader.load_seq_lpad_validpipe(
            dataset, cfg.maxlen, 
            cfg.NUM_PADS, padding_value=0,
            batch_size=100, ranking=cfg.ranking
        )
        testpipe = freerec.data.dataloader.load_seq_lpad_testpipe(
            dataset, cfg.maxlen, 
            cfg.NUM_PADS, padding_value=0,
            batch_size=100, ranking=cfg.ranking
        )
    else:
        raise NotImplementedError()

    return dataset, trainpipe, validpipe, testpipe