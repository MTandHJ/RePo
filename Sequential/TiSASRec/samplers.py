

from typing import Iterable

import numpy as np
import torchdata.datapipes as dp

import freerec
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID, POSITIVE, UNSEEN, SEEN
from freerec.utils import timemeter


@dp.functional_datapipe("ti_seq_train_uniform_sampling_")
class TiSASRecTrainSampler(freerec.data.postprocessing.sampler.SeqTrainUniformSampler):

    @timemeter
    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet
            The dataset object that contains field objects.
        """
        self.posItems = [[] for _ in range(self.User.count)]
        self.posTimes = [[] for _ in range(self.User.count)]
        self.negative_pool = self._sample_from_all(dataset.datasize)

        for chunk in dataset.train():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )

        self.posItems = [tuple(items) for items in self.posItems]

    def __iter__(self):
        for user, seq in self.source:
            if self._check(seq):
                seen = seq[:-1]
                times = self.posTimes[user]
                positives = seq[self.marker:]
                negatives = self._sample_neg(user, positives)
                yield [user, seen, times[:-1], positives, self._sample_neg(user, negatives)]


@dp.functional_datapipe("ti_seq_valid_sampling_")
class TiSASRecValidSampler(freerec.data.postprocessing.sampler.SeqValidSampler):

    @timemeter
    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet
            The dataset object that contains field objects.
        """
        self.posItems = [[] for _ in range(self.User.count)]
        self.posTimes = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )

        self.negItems = self.listmap(
            self._sample_negs, self.posItems
        )

        for chunk in dataset.valid():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        self.posItems = [tuple(items) for items in self.posItems]

    def __iter__(self):
        for user in self.source:
            posItems = self.posItems[user]
            if self._check(posItems):
                times = self.posTimes[user]
                yield [user, posItems[:-1], times[:-1], posItems[-1:] + self.negItems[user]]


@dp.functional_datapipe("ti_seq_test_sampling_")
class TiSASRecTestSampler(freerec.data.postprocessing.sampler.SeqTestSampler):

    @timemeter
    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet
            The dataset object that contains field objects.
        """
        self.posItems = [[] for _ in range(self.User.count)]
        self.posTimes = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )

        self.negItems = self.listmap(
            self._sample_negs, self.posItems
        )

        for chunk in dataset.test():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        self.posItems = [tuple(items) for items in self.posItems]

    def __iter__(self):
        for user in self.source:
            posItems = self.posItems[user]
            if self._check(posItems):
                times = self.posTimes[user]
                yield [user, posItems[:-1], times[:-1], posItems[-1:] + self.negItems[user]]


@dp.functional_datapipe("ti_seq_valid_yielding_")
class TiSASRecValidYielder(freerec.data.postprocessing.sampler.SeqValidYielder):

    @timemeter
    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet
            The dataset object that contains field objects.
        """
        self.posItems = [[] for _ in range(self.User.count)]
        self.posTimes = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        self.posItems = [tuple(items) for items in self.posItems]

    def __iter__(self):
        for user in self.source:
            posItems = self.posItems[user]
            if self._check(posItems):
                times = self.posTimes[user]
                # (user, seqs, times, unseen, seen)
                yield [user, posItems[:-1], times[:-1], posItems[-1:], posItems[:-1]]


@dp.functional_datapipe("ti_seq_test_yielding_")
class TiSASRecTestYielder(freerec.data.postprocessing.sampler.SeqTestYielder):

    @timemeter
    def prepare(self, dataset: freerec.data.datasets.RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet
            The dataset object that contains field objects.
        """
        self.posItems = [[] for _ in range(self.User.count)]
        self.posTimes = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        for chunk in dataset.test():
            self.listmap(
                lambda user, item, time: (self.posItems[user].append(item), self.posTimes[user].append(int(time))),
                chunk[USER, ID], chunk[ITEM, ID], chunk[TIMESTAMP]
            )
        self.posItems = [tuple(items) for items in self.posItems]

    def __iter__(self):
        for user in self.source:
            posItems = self.posItems[user]
            if self._check(posItems):
                times = self.posTimes[user]
                # (user, seqs, times, unseen, seen)
                yield [user, posItems[:-1], times[:-1], posItems[-1:], posItems[:-1]]


@dp.functional_datapipe("time2matrix_")
class TimeMapper(freerec.data.postprocessing.row.RowMapper):

    def __init__(self, source_dp: dp.iter.IterableWrapper, indices: Iterable[int], time_span: int = 256):
        super().__init__(source_dp, self._time2matrix, indices)
        self.time_span = time_span

    def _time2matrix(self, times: Iterable) -> Iterable:
        times = np.array(times, dtype=np.float32).reshape((-1, 1))
        min_time = np.unique(times)
        min_time = max(1, np.min(min_time[1:] - min_time[:-1]).item())
        matrix = (np.abs(times - times.T) // min_time).astype(int)
        return np.clip(matrix, 0, self.time_span).tolist()