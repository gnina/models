import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):
    def __init__(self, input_size, config, num_dense_blocks=3, num_dense_filters=16, num_dense_convs=4):
        super().__init__()
    
        self.modules = []
        in_feats = input_size[0]
        out_feats = 32

        self.dropout = nn.Dropout(p=config.dropout)

        self.data_enc_init_pool = nn.MaxPool3d(2, stride=2)
        self.data_enc_init_conv = nn.Conv3d(in_feats, out_feats, 3, padding=1)

        for idx in range(num_dense_blocks - 1):
            in_feats = out_feats
            dense_block = DenseBlock(in_feats, idx, num_dense_convs, num_dense_filters)
            self.add_module(f"dense_block_{idx}", dense_block)
            self.modules.append(dense_block)
            out_feats = num_dense_convs * num_dense_filters + in_feats
            
            bottleneck = nn.Conv3d(out_feats, out_feats, 1, padding=0)
            self.add_module(f"data_enc_level{idx}_bottleneck", bottleneck)
            self.modules.append(bottleneck)

            max_pool = nn.MaxPool3d(2, stride=2)
            self.add_module(f"data_enc_level{idx+1}_pool", max_pool)
            self.modules.append(max_pool)

        in_feats = out_feats
        dense_block = DenseBlock(in_feats, num_dense_blocks-1, num_dense_convs, num_dense_filters)
        out_feats = num_dense_convs * num_dense_filters + in_feats
        self.add_module(f"dense_block_{num_dense_blocks-1}", dense_block)
        self.modules.append(dense_block)

        self.fc = nn.Linear(out_feats,1)

    def forward_one(self,x):
        x = self.data_enc_init_pool(x) 
        x = F.relu(self.data_enc_init_conv(x))
        for module in self.modules:
            x = module(x)
            if isinstance(module, nn.Conv3d):
                x = F.relu(x)
        
        # Computing the global pooling, kernel is the size of the input
        B, C, D, H, W, = x.size()
        x = F.max_pool3d(x, (D, H, W)).view(B, C)

        return self.fc(x)

class DenseBlock(nn.Module):
    def __init__(self, input_feats, level, convolutions=4, num_dense_filters=16):
        super().__init__()

        self.modules = []
        current_feats = input_feats
        for idx in range(convolutions):
            bn = nn.BatchNorm3d(current_feats)
            self.add_module(f"data_enc_level{level}_batchnorm_conv{idx}", bn)
            self.modules.append(bn)

            conv = nn.Conv3d(current_feats, num_dense_filters, kernel_size=3, padding=1)
            self.add_module(f"data_enc_level{level}_conv{idx}", conv)
            self.modules.append(conv)
            current_feats += num_dense_filters

    def forward(self, x):
        previous = []
        previous.append(x)
        for module in self.modules:
            x = module(x)
            if isinstance(module, nn.Conv3d):
                x = F.relu(x)
                previous.append(x)
                x = torch.cat(previous, 1)
        return x
