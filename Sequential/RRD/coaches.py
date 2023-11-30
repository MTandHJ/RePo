

import freerec


class CoachForBPRMF(freerec.launcher.GenCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, positives, negatives = [col.to(self.device) for col in data]
            logits_s, logits_i, logits_u = self.model.predict(users, positives, negatives.unsqueeze(1))
            loss = self.criterion(logits_s, logits_i, logits_u)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class CoachForGRU4Rec(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            logits_s, logits_i, logits_u = self.model.predict(seqs, positives, negatives)
            loss = self.criterion(logits_s, logits_i, logits_u)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


class CoachForSASRec(freerec.launcher.SeqCoach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs, positives, negatives = [col.to(self.device) for col in data]
            logits_s, logits_i, logits_u = self.model.predict(seqs, positives, negatives)
            indices = positives.flatten() != 0
            logits_s = logits_s[indices]
            logits_i = logits_i[indices]
            logits_u = logits_u[indices]
            loss = self.criterion(logits_s, logits_i, logits_u)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])