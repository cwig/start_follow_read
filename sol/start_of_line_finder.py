import torch
from torch.autograd import Variable
from torch import nn
import vgg

class StartOfLineFinder(nn.Module):
    def __init__(self, base_0, base_1):
        super(StartOfLineFinder, self).__init__()

        self.cnn = vgg.vgg11()
        self.base_0 = base_0
        self.base_1 = base_1

    def forward(self, img):
        y = self.cnn(img)

        priors_0 = Variable(torch.arange(0,y.size(2)).type_as(img.data), requires_grad=False)[None,:,None]
        priors_0 = (priors_0 + 0.5) * self.base_0
        priors_0 = priors_0.expand(y.size(0), priors_0.size(1), y.size(3))
        priors_0 = priors_0[:,None,:,:]

        priors_1 = Variable(torch.arange(0,y.size(3)).type_as(img.data), requires_grad=False)[None,None,:]
        priors_1 = (priors_1 + 0.5) * self.base_1
        priors_1 = priors_1.expand(y.size(0), y.size(2), priors_1.size(2))
        priors_1 = priors_1[:,None,:,:]

        predictions = torch.cat([
            torch.sigmoid(y[:,0:1,:,:]),
            y[:,1:2,:,:] + priors_0,
            y[:,2:3,:,:] + priors_1,
            y[:,3:4,:,:],
            y[:,4:5,:,:]
        ], dim=1)

        predictions = predictions.transpose(1,3).contiguous()
        predictions = predictions.view(predictions.size(0),-1,5)

        return predictions
