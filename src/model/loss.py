import torch

EPS = 1e-8


def mask_output(loss_function):

    class MaskOutput(loss_function):
        def __init__(self, size_average=None, reduce=None, reduction: str = 'none', silent_weight=1, *args, **kwargs):
            super(MaskOutput, self).__init__(size_average, reduce, reduction=reduction, **kwargs)
            self.silent_weight = silent_weight

        def forward(self, scores, labels, mask=None):
            if mask is None:
                mask = torch.ones_like(labels)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            loss = super(MaskOutput, self).forward(scores, labels)
            loss[labels == 0] *= self.silent_weight
            loss = loss.reshape(mask.shape)
            loss = (loss*mask).sum(-1)/mask.sum(-1)
            loss = loss.mean()
            return loss

    return MaskOutput


masked_cross_entropy_loss_wl = mask_output(torch.nn.BCEWithLogitsLoss)
masked_cross_entropy_loss = mask_output(torch.nn.BCELoss)
cross_entropy_loss_wl = torch.nn.BCEWithLogitsLoss

if __name__ == "__main__":
    pass

