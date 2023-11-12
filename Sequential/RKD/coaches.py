

import freerec


class CoachForBPRMF(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            userFeats_s, itemFeats_s, userFeats_t, itemFeats_t = self.model.predict(users, positives, negatives)
            loss = self.criterion(
                userFeats_s, itemFeats_s,
                userFeats_t, itemFeats_t
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class CoachForGRU4Rec(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            userFeats_s, itemFeats_s, userFeats_t, itemFeats_t = self.model.predict(seqs, positives, negatives)
            loss = self.criterion(
                userFeats_s, itemFeats_s,
                userFeats_t, itemFeats_t
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class CoachForSASRec(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            userFeats_s, itemFeats_s, userFeats_t, itemFeats_t = self.model.predict(seqs, positives, negatives)
            # xxxxFeats: (B * S, 1/2, D) 
            indices = positives.flatten() != 0
            userFeats_s = userFeats_s[indices]
            itemFeats_s = itemFeats_s[indices]
            userFeats_t = userFeats_t[indices]
            itemFeats_t = itemFeats_t[indices]

            loss = self.criterion(
                userFeats_s, itemFeats_s,
                userFeats_t, itemFeats_t
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])