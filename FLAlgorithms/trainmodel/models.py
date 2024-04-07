import torch
from torch import nn
import torch.distributions as tdist
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, remove_spectral_norm
import numpy as np
from torchvision.models import vgg16, resnet50, mobilenet_v2, resnet18

## Initial
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


## DNNs
class VGG(torch.nn.Module): #API
    def __init__(self, output):
        super(VGG, self).__init__()
        self.vgg=vgg16(pretrained=True).features
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(512, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 10))
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        x = self.vgg(input)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits 
        
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layers = nn.Sequential(
                        nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size = 2, stride = 2)),
                        nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2)),
                        nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(kernel_size = 2, stride = 2)),
                        nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(kernel_size = 2, stride = 2)),
                        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU()),
                        nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(kernel_size = 2, stride = 2)))
 
        self.fc = nn.Sequential(nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        self.fc1 = nn.Sequential(nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        self.fc2= nn.Sequential(nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.layers(x)      
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out        
        
    def forward_feature(self, feature, idx=9):
        for i in range(idx+1, 13):
            out = self.layers[i](feature)
            feature = out    
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
    def get_feature(self, x, idx=9):
        with torch.no_grad():
            feature = x
            for i in range(idx+1):
                feature = self.layers[i](feature)
            return feature
        
        
class RESNET(torch.nn.Module): #API
    def __init__(self, output):
        super(RESNET, self).__init__()
        self.resnet=resnet18(pretrained=False, num_classes=10)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = torch.nn.Identity()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        print(input.shape)
        x = self.resnet(input)
        return x 
               
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out     
 
class ResNet18(nn.Module):
    def __init__(self, num_classes=10, img_channels=3,num_layers=18, block=BasicBlock):
        super(ResNet18, self).__init__()
        layers = [2, 2, 2, 2]
        self.expansion = 1
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=self.in_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.Sequential(self._make_layer(block, 64, layers[0], stride=1),
                                    self._make_layer(block, 128, layers[1], stride=2),
                                    self._make_layer(block, 256, layers[2], stride=2),
                                    self._make_layer(block, 512, layers[3], stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
        
    def _make_layer(self, block,out_channels,blocks,stride):
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
        
    def forward_feature(self, x, idx=9):
        for i in range(idx, 4):
            x = self.layers[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
    def get_feature(self, x, idx=9):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            if idx > 0: #layer 0 is the feature from first conv
                for i in range(idx):
                    x = self.layers[i](x)
            return x 

            
class MOBNET(torch.nn.Module):
    def __init__(self, output):
        super(MOBNET, self).__init__()
        self.mobilenet=mobilenet_v2(pretrained=False, num_classes=10)
        
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        x = self.mobilenet(input)
        
        return x 

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_feature(self, x, idx=9):
        feature = F.relu(self.bn1(self.conv1(x)))
        if idx <=16:
            for i in range(idx+1):
                feature = self.layers[i](feature)
            return feature
            
        else:   #idx 17 --> linear layer
            feature = self.layers(feature)
            out = F.relu(self.bn2(self.conv2(feature)))
            out = F.avg_pool2d(out, 4) ##
            out = out.view(out.size(0), -1)
            return out

    def forward_feature(self, feature, idx=9):
        if idx <= 16:
            for i in range(idx+1, 17):
                out = self.layers[i](feature)
                feature = out    
            out = F.relu(self.bn2(self.conv2(feature)))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            #out = feature.view(feature.size(0), -1)
            out = self.linear(out)
        return out
 
class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width=256):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.hiddens = nn.ModuleList([nn.Linear(mlp_width, mlp_width) for _ in range(1)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        x = self.output(x)
        return x
 
 
class MinimalDecoder(nn.Module):
    def __init__(self, input_nc, output_nc=3, input_dim=None, output_dim=None):
        super(MinimalDecoder, self).__init__()
        model = [nn.Conv2d(input_nc, 32, kernel_size=1)] #(16,32,32) -> (3,32,32)
        model += [nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)]
        model += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)] 
        model += [nn.Conv2d(32, 3, kernel_size=1, stride=1)] 
        self.m = nn.Sequential(*model)

    def forward(self, x):
        for idx, m in enumerate(self.m):
            x = m(x)
        return x

class FeatureClF(nn.Module):
    def __init__(self, input_nc=16, output_dim=2):
        super(FeatureClF, self).__init__()
        model = [nn.Conv2d(input_nc, 32, kernel_size=1)]  
        model += [nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)] 
        model += [nn.Conv2d(32, 8, kernel_size=1, stride=1)] 
        self.m = nn.Sequential(*model)
        self.linear = nn.Linear(2048, output_dim) #3x16x16
    def forward(self, x):
        for idx, m in enumerate(self.m):
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class VIT(torch.nn.Module): #API
    def __init__(self, output):
        super(VIT, self).__init__()
        self.vit=vit_b_16(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(1000, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(True),
                                        nn.Linear(512, 10))
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        x = self.vit(input)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

## https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    #print('batch', B, x.shape)
    return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, num_heads=8, dropout=0.5):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
        
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10, patch_size=4,num_patches=64, embed_dim=256,hidden_dim=512,num_channels=3,num_heads=8,num_layers=10,dropout=0.2):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size
        self.num_layers = num_layers
        
        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)))
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
        
    def forward_feature(self, feature, idx=2):
    
        feature = feature.transpose(0, 1)
        for i in range(idx+1, self.num_layers):
            out = self.transformer[i](feature)
            feature = out    
        cls = out[0]
        out = self.mlp_head(cls)
        return out
        
    def get_feature(self, x, idx=9):
        with torch.no_grad():
            # Preprocess input
            x = img_to_patch(x, self.patch_size)
            B, T, _ = x.shape
            x = self.input_layer(x)

            # Add CLS token and positional encoding
            cls_token = self.cls_token.repeat(B, 1, 1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.pos_embedding[:, : T + 1]

            # Apply Transforrmer
            x = self.dropout(x)
            feature = x.transpose(0, 1)
            for i in range(idx+1):
                feature = self.transformer[i](feature)
               
            feature = feature.transpose(0, 1) ## reverse transpose
            return feature        
        


import torch.nn.functional as F
from torch.nn import init
from torch import nn

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (torch.nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(self._maker_layer0(), self._maker_layer1(),self._maker_layer2(),self._maker_layer3())
        # Linear Classifier
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = torch.nn.Linear(in_features=64, out_features=10)
        

    def _maker_layer0(self):
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        conv1 = torch.nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        relu1 = torch.nn.ReLU()
        bn1 = torch.nn.BatchNorm2d(8)
        init.kaiming_normal_(conv1.weight, a=0.1)
        conv1.bias.data.zero_()
        return nn.Sequential(conv1, relu1, bn1)

    def _maker_layer1(self):
        # Second Convolution Block
        conv2 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        relu2 = torch.nn.ReLU()
        bn2 = torch.nn.BatchNorm2d(16)
        init.kaiming_normal_(conv2.weight, a=0.1)
        conv2.bias.data.zero_()
        return nn.Sequential(conv2, relu2, bn2)

    def _maker_layer2(self):
        # Second Convolution Block
        conv3 = torch.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        relu3 = torch.nn.ReLU()
        bn3 = torch.nn.BatchNorm2d(32)
        init.kaiming_normal_(conv3.weight, a=0.1)
        conv3.bias.data.zero_()
        return nn.Sequential(conv3, relu3, bn3)

    def _maker_layer3(self):
        # Second Convolution Block
        conv4 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        relu4 = torch.nn.ReLU()
        bn4 = torch.nn.BatchNorm2d(64)
        init.kaiming_normal_(conv4.weight, a=0.1)
        conv4.bias.data.zero_()
        return nn.Sequential(conv4, relu4, bn4)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.layers(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
        
        
    def forward_feature(self, feature, idx=1):
        if idx < 3:
            for i in range(idx+1, 4):
                x = self.layers[i](feature)
                feature = x    
            x = self.ap(x)
            x = x.view(x.shape[0], -1)
            out = self.lin(x)
        else:
            feature = self.ap(feature)
            out = feature.view(feature.size(0), -1)
            out = self.lin(out)
        return out
        
    def get_feature(self, x, idx=1):
        feature = x
        if idx <=3:
            for i in range(idx+1):
                feature = self.layers[i](feature)
            return feature


# ----------------------------
# Sensory Data Classification Model
# ----------------------------
class HARNet (torch.nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self, num_classes=6):
        super().__init__()
        self.layers = nn.Sequential(self._maker_lencoder0(),self._maker_lencoder1(), self._maker_lencoder2(),self._maker_lencoder3())
        # Linear Classifier
        self.lin = torch.nn.Linear(in_features=8448, out_features=num_classes)
        

    def _maker_lencoder0(self):
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,1),stride=(3,1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
            )
    
    def _maker_lencoder1(self):
        return nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(6,1),stride=(2,1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
            )

    def _maker_lencoder2(self):
        return nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(6,1),stride=(2,1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
            )

    def _maker_lencoder3(self):
        return  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(6,1),stride=(2,1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d((4, 1), stride=(1, 1),padding=(1,0)),
            )




    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.layers(x)
        x = x.contiguous().view(x.size(0), -1)
        # Linear layer
        x = self.lin(x)

        # Final output
        return x
        
        
    def forward_feature(self, feature, idx=1):
        if idx < 2:
            for i in range(idx+1, 4):
                x = self.layers[i](feature)
                feature = x    
            feature = x.contiguous().view(x.size(0), -1)
            out = self.lin(feature)
        else:
            feature = feature.contiguous().view(feature.size(0), -1)
            out = self.lin(feature)
        return out
        
    def get_feature(self, x, idx=1):
        feature = x
        if idx <=2:
            for i in range(idx+1):
                feature = self.layers[i](feature)
            return feature

