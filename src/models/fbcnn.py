import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class QFAttentionBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )

    def forward(self, x, gamma, beta):
        res = self.body(x)
        res = gamma * res + beta
        return x + res


class FlexibleController(nn.Module):
    def __init__(self, nc=(64, 128, 256)):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )

        self.to_gamma_1 = nn.Sequential(nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = nn.Sequential(nn.Linear(512, nc[0]), nn.Tanh())
        self.to_gamma_2 = nn.Sequential(nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = nn.Sequential(nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_3 = nn.Sequential(nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = nn.Sequential(nn.Linear(512, nc[2]), nn.Tanh())

    def forward(self, q):
        emb = self.mlp(q)

        def parameters(gamma, beta):
            return gamma(emb)[..., None, None], beta(emb)[..., None, None]

        return (
            parameters(self.to_gamma_1, self.to_beta_1),
            parameters(self.to_gamma_2, self.to_beta_2),
            parameters(self.to_gamma_3, self.to_beta_3),
        )


class FBCNN(nn.Module):
    def __init__(
        self,
        in_nc=3,
        out_nc=3,
        nc=(64, 128, 256, 512),
        clamp_output=True,
        quality_loss_weight=0.1,
    ):
        super().__init__()
        self.clamp_output = clamp_output
        self.quality_loss_weight = quality_loss_weight

        self.head = nn.Conv2d(in_nc, nc[0], 3, 1, 1)

        self.down1 = nn.Sequential(
            *[ResBlock(nc[0]) for _ in range(4)],
            nn.Conv2d(nc[0], nc[1], 2, 2, 0),
        )
        self.down2 = nn.Sequential(
            *[ResBlock(nc[1]) for _ in range(4)],
            nn.Conv2d(nc[1], nc[2], 2, 2, 0),
        )
        self.down3 = nn.Sequential(
            *[ResBlock(nc[2]) for _ in range(4)],
            nn.Conv2d(nc[2], nc[3], 2, 2, 0),
        )

        self.body_encoder = nn.Sequential(*[ResBlock(nc[3]) for _ in range(4)])
        self.body_decoder = nn.Sequential(*[ResBlock(nc[3]) for _ in range(4)])

        self.qf_branch = nn.Sequential(
            *[ResBlock(nc[3]) for _ in range(4)],
            nn.AdaptiveAvgPool2d(1),
        )

        self.qf_predictor = nn.Sequential(
            nn.Linear(nc[3], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.controller = FlexibleController(nc[:3])

        self.up3 = nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0)
        self.rec_scale3 = nn.ModuleList([QFAttentionBlock(nc[2]) for _ in range(4)])

        self.up2 = nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0)
        self.rec_scale2 = nn.ModuleList([QFAttentionBlock(nc[1]) for _ in range(4)])

        self.up1 = nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0)
        self.rec_scale1 = nn.ModuleList([QFAttentionBlock(nc[0]) for _ in range(4)])

        self.tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1)

    def forward(self, x, external_q=None):
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        feat1 = self.head(x)
        feat2 = self.down1(feat1)
        feat3 = self.down2(feat2)
        feat4 = self.down3(feat3)

        x = self.body_encoder(feat4)
        qf_feat = self.qf_branch(x).flatten(1)
        q_pred = self.qf_predictor(qf_feat)
        x = self.body_decoder(x)
        x = x + feat4

        q = external_q if external_q is not None else q_pred
        (g1, b1), (g2, b2), (g3, b3) = self.controller(q)

        x = self.up3(x)
        for block in self.rec_scale3:
            x = block(x, g3, b3)
        x = x + feat3

        x = self.up2(x)
        for block in self.rec_scale2:
            x = block(x, g2, b2)
        x = x + feat2

        x = self.up1(x)
        for block in self.rec_scale1:
            x = block(x, g1, b1)
        x = x + feat1

        x = self.tail(x)
        x = x[..., :h, :w]
        if self.clamp_output:
            x = torch.clamp(x, 0.0, 1.0)
        return x, q_pred

    def compute_loss(self, degraded, clean, quality):
        restored, predicted_quality = self(degraded)
        reconstruction = F.l1_loss(restored, clean)
        quality_loss = F.l1_loss(predicted_quality, quality)
        return {
            "loss": reconstruction + self.quality_loss_weight * quality_loss,
            "reconstruction": reconstruction,
            "quality": quality_loss,
        }


if __name__ == "__main__":
    model = FBCNN().to("cuda")

    x = torch.randn((4, 3, 128, 128), device="cuda")
    out, q = model(x)
    print(out.shape, q.shape)
