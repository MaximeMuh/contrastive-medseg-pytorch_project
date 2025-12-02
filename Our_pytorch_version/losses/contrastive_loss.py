import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalContrastiveLoss(nn.Module):
    """
    Local contrastive loss for pixel embeddings.

    We follow the idea from the paper:
    - positives = one pixel vs the mean feature of its own class
    - negatives = mean features of the other classes (including background)
    - uses a stable InfoNCE loss (always >= 0)
    - still works when K = 2 (background + one class)
    - avoids having loss = 0 when we do not have enough foreground classes
    """

    def __init__(self, num_classes, temperature=0.2, num_pos_samples=1,
                 inter_image=False):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.num_pos_samples = num_pos_samples
        self.inter_image = inter_image

    @staticmethod
    def _class_mean(emb, mask_bool):
        """
        Compute mean feature for one class.

        emb: [C,H,W], mask_bool: [H,W]
        return: [C]
        """
        if mask_bool.sum() == 0:
            return emb.new_zeros(emb.size(0))
        return emb[:, mask_bool].mean(dim=1)

    def _sim(self, a, b):
        """Cosine similarity scaled by temperature."""
        return F.cosine_similarity(a, b, dim=0) / self.temperature

    def forward(self, embeddings, masks_onehot):
        """
        Compute local contrastive loss.

        embeddings : [B, E, H, W]  (pixel embeddings)
        masks_onehot : [B, K, H, W]  (one-hot labels for each pixel)
        """
        device = embeddings.device
        B, E, H, W = embeddings.shape
        K = self.num_classes

        num_pairs = B // 2
        total_loss = 0.0
        total_count = 0

        for p in range(num_pairs):
            i1, i2 = p, p + num_pairs
            emb1, emb2 = embeddings[i1], embeddings[i2]   # [E,H,W]
            m1, m2 = masks_onehot[i1], masks_onehot[i2]  # [K,H,W]

            for cls in range(K):  
                mask1 = m1[cls].bool()
                mask2 = m2[cls].bool()

                if not (mask1.any() and mask2.any()):
                    continue

                mean1 = self._class_mean(emb1, mask1)
                mean2 = self._class_mean(emb2, mask2)

                neg_classes = [c for c in range(K) if c != cls]

                if len(neg_classes) == 0:
                    continue

                pos_idx1 = torch.nonzero(mask1, as_tuple=False)
                pos_idx2 = torch.nonzero(mask2, as_tuple=False)

                n_pos = min(self.num_pos_samples,
                            pos_idx1.size(0), pos_idx2.size(0))
                if n_pos == 0:
                    continue

                for _ in range(n_pos):

                    r1 = pos_idx1[torch.randint(0, pos_idx1.size(0), (1,))]
                    r2 = pos_idx2[torch.randint(0, pos_idx2.size(0), (1,))]
                    z1 = emb1[:, r1[0, 0], r1[0, 1]]
                    z2 = emb2[:, r2[0, 0], r2[0, 1]]

                    # POSITIVES: pixel vs mean feature of the same class
                    s_pos1 = self._sim(z1, mean1)
                    s_pos2 = self._sim(z2, mean2)

                    # NEGATIVES: class means from all the other classes
                    negs1, negs2 = [], []
                    for nc in neg_classes:
                        neg_mask1 = m1[nc].bool()
                        neg_mask2 = m2[nc].bool()

                        if neg_mask1.any():
                            negs1.append(self._sim(z1, self._class_mean(emb1, neg_mask1)))
                        if neg_mask2.any():
                            negs2.append(self._sim(z2, self._class_mean(emb2, neg_mask2)))

                    if len(negs1) == 0:
                        # fallback if we do not have any negative in this image
                        negs1.append(s_pos1 - 1.0)  
                    if len(negs2) == 0:
                        negs2.append(s_pos2 - 1.0)

                    # InfoNCE loss for each pixel-class pair
                    den1 = torch.logsumexp(torch.stack([s_pos1] + negs1), dim=0)
                    loss1 = -(s_pos1 - den1)

                    den2 = torch.logsumexp(torch.stack([s_pos2] + negs2), dim=0)
                    loss2 = -(s_pos2 - den2)

                    total_loss += 0.5 * (loss1 + loss2)
                    total_count += 1

        if total_count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / total_count