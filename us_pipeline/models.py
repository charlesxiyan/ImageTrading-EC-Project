"""One five-minute model: CNN78u, with explicit configurable convolutional depth."""
from torch import nn


class CNN78u(nn.Module):
    """A 78-bar image classifier; 78 is input intervals, not hidden neurons.

    Four blocks are a working hypothesis, not a validated performance improvement.
    The adaptive head retains coarse price/time locations without a huge flatten FC.
    """
    def __init__(self,channels=(32,64,128,256),pool_grid=4,dropout=.5):
        super().__init__()
        if not channels or len(channels)>4 or any(int(c)!=c or c<=0 for c in channels):
            raise ValueError('Use one to four positive integer convolutional widths')
        if not 1<=pool_grid<=14 or not 0<=dropout<1:
            raise ValueError('Invalid adaptive grid or dropout')
        layers=[];incoming=1
        for outgoing in channels:
            layers.extend([nn.Conv2d(incoming,outgoing,(5,3),padding=(2,1)),
                nn.BatchNorm2d(outgoing),nn.ReLU(),nn.MaxPool2d(2,ceil_mode=True)])
            incoming=outgoing
        self.blocks=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d((pool_grid,pool_grid)),
            nn.Flatten(),nn.Dropout(dropout),nn.Linear(channels[-1]*pool_grid*pool_grid,2))
        self.channels=tuple(channels)
        self.pool_grid=pool_grid
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias,.01)

    @property
    def convolutional_receptive_field(self):
        # Interior theoretical RF before adaptive pooling; head combines all regions.
        height=width=jump=1
        for _ in self.channels:
            height+=4*jump;width+=2*jump
            height+=jump;width+=jump
            jump*=2
        return height,width

    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(1,234,234):
            raise ValueError('CNN78u expects [batch,1,234,234] from 78 five-minute bars')
        return self.head(self.blocks(x))


def build_model(config):
    settings=config['model']
    if config['lookback']!=78 or config['bar_minutes']!=5 or settings['name']!='CNN78u':
        raise ValueError('Only the five-minute CNN78u configuration is active')
    return CNN78u(channels=tuple(settings['channels']),pool_grid=settings['pool_grid'],
                  dropout=settings['dropout'])
