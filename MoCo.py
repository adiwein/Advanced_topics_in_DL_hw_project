import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional


class MoCo(torch.nn.Module):
    def __init__(self, args):
        super(MoCo, self).__init__()

        self.args = args
        self.query_encoder = models.resnet50(num_classes=self.args.dim)
        self.key_encoder = models.resnet50(num_classes=self.args.dim)
        if args.mlp:
            in_features = self.query_encoder.fc.weight.shape[1]
            mlp = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=in_features, out_features=args.dim, bias=True)
            )
            self.key_encoder.fc = mlp
            self.query_encoder.fc = mlp

        for key_param, query_param in zip(self.key_encoder.parameters(), self.query_encoder.parameters()):
            # key_param = query_param.data.clone()
            key_param.requires_grad = False

        queue_init = functional.normalize(torch.randn(args.dim, args.k), dim=0)
        self.register_buffer('queue', queue_init)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, x_q, x_k, ):
        q = self.query_encoder(x_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self.momentum_update_key_encoder()

            k = self.key_encoder(x_k)
            k = nn.functional.normalize(k, dim=1)
            if not q.shape == torch.Size([self.args.batch_size, self.args.dim]):
                print(q.shape)
                logits = None
                labels = None
                return logits, labels

        l_pos = torch.matmul(q.view(self.args.batch_size, 1, self.args.dim),
                             k.view(self.args.batch_size, self.args.dim, 1)).squeeze(1)
        l_neg = torch.mm(q.view(self.args.batch_size, self.args.dim), self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], 1)
        logits /= self.args.t
        labels = torch.zeros(self.args.batch_size, dtype=torch.long)

        self.dequeue_and_enqueue(k)

        return logits, labels

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = self.args.m * param_k.data + (1 - self.args.m) * param_q.data

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.args.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.args.k  # move pointer

        self.queue_ptr[0] = ptr



