"""
yolo_fastest_to_pytorch.py

Convert YOLO-Fastest Darknet models to PyTorch by parsing given config and
weights file.

https://github.com/dog-qiuqiu/Yolo-Fastest

Full disclosure: this was written almost entirely by Claude AI. If you find any
issues with the script, please let me know.

License: Apache-2.0
"""

import torch
import torch.nn as nn
import numpy as np


def parse_cfg(cfg_path):
    """Parse Darknet config file into list of layer dicts."""
    with open(cfg_path, 'r') as f:
        lines = f.read().split('\n')
    
    blocks = []
    block = {}
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            if block:
                blocks.append(block)
            block = {'type': line[1:-1]}
        else:
            if '=' in line:
                key, value = line.split('=', 1)
                block[key.strip()] = value.strip()
    
    if block:
        blocks.append(block)
    
    return blocks


class ConvBN(nn.Module):
    """Convolution + BatchNorm + Activation."""
    def __init__(self, in_ch, out_ch, kernel, stride, pad, groups=1, activation='leaky'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        
        if activation == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'linear':
            self.act = nn.Identity()
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvNoBN(nn.Module):
    """Convolution without BatchNorm."""
    def __init__(self, in_ch, out_ch, kernel, stride, pad, groups=1, activation='linear'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, groups=groups, bias=True)
        
        if activation == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'linear':
            self.act = nn.Identity()
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.act(self.conv(x))


class YoloFastest(nn.Module):
    """YOLO-Fastest model built from config."""
    
    def __init__(self, cfg_path, input_size=192, debug=False):
        super().__init__()
        self.blocks = parse_cfg(cfg_path)
        self.input_size = input_size
        self.debug = debug
        
        # Build network
        self.module_list = nn.ModuleList()
        
        # Track output filters for each layer
        # output_filters[i] corresponds to the output of module_list[i-1]
        # output_filters[0] = input channels (3 for RGB)
        self.output_filters = [3]  # Start with input
        
        layer_idx = 0  # Counts non-net blocks
        
        for block in self.blocks:
            if block['type'] == 'net':
                continue
            
            # Get input channels from previous layer
            in_ch = self.output_filters[-1]
            
            if block['type'] == 'convolutional':
                out_ch = int(block['filters'])
                groups = int(block.get('groups', 1))
                
                # For depthwise conv (groups > 1), input channels must equal groups
                if groups > 1:
                    if in_ch != groups and self.debug:
                        print(f"Layer {layer_idx}: depthwise conv expects {groups} channels, got {in_ch}")
                    in_ch = groups  # Use groups as in_ch for depthwise
                
                module = self._make_conv(block, in_ch, out_ch)
                self.output_filters.append(out_ch)
                
                if self.debug:
                    print(f"Layer {layer_idx}: conv in={in_ch} out={out_ch} groups={groups}")
                
            elif block['type'] == 'shortcut':
                from_idx = int(block['from'])
                actual_idx = len(self.output_filters) - 1 + from_idx
                self.output_filters.append(self.output_filters[-1])
                
                if self.debug:
                    print(f"Layer {layer_idx}: shortcut from={from_idx} (abs={actual_idx}) out={self.output_filters[-1]}")
                
                module = nn.Identity()
                
            elif block['type'] == 'route':
                layers_str = block['layers']
                layers = [int(x.strip()) for x in layers_str.split(',')]
                
                out_ch = 0
                resolved_indices = []
                for l in layers:
                    if l > 0:
                        idx = l + 1
                    else:
                        idx = len(self.output_filters) + l
                    resolved_indices.append(idx)
                    out_ch += self.output_filters[idx]
                
                self.output_filters.append(out_ch)
                
                if self.debug:
                    print(f"Layer {layer_idx}: route layers={layers} resolved={resolved_indices} out={out_ch}")
                
                module = nn.Identity()
                
            elif block['type'] == 'maxpool':
                size = int(block['size'])
                stride = int(block['stride'])
                pad = (size - 1) // 2
                module = nn.MaxPool2d(size, stride, pad)
                self.output_filters.append(self.output_filters[-1])
                
                if self.debug:
                    print(f"Layer {layer_idx}: maxpool out={self.output_filters[-1]}")
                
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                module = nn.Upsample(scale_factor=stride, mode='nearest')
                self.output_filters.append(self.output_filters[-1])
                
                if self.debug:
                    print(f"Layer {layer_idx}: upsample out={self.output_filters[-1]}")
                
            elif block['type'] == 'yolo':
                module = nn.Identity()
                self.output_filters.append(self.output_filters[-1])
                
                if self.debug:
                    print(f"Layer {layer_idx}: yolo out={self.output_filters[-1]}")
                
            elif block['type'] == 'dropout':
                module = nn.Identity()
                self.output_filters.append(self.output_filters[-1])
                
                if self.debug:
                    print(f"Layer {layer_idx}: dropout out={self.output_filters[-1]}")
                
            else:
                print(f"Unknown layer type: {block['type']}")
                module = nn.Identity()
                self.output_filters.append(self.output_filters[-1])
            
            self.module_list.append(module)
            layer_idx += 1
        
        # Store route info for forward pass
        self._precompute_route_info()
    
    def _precompute_route_info(self):
        """Precompute route layer indices for forward pass."""
        self.route_info = {}
        self.shortcut_info = {}
        
        layer_idx = 0
        for block in self.blocks:
            if block['type'] == 'net':
                continue
            
            if block['type'] == 'route':
                layers = [int(x.strip()) for x in block['layers'].split(',')]
                resolved = []
                for l in layers:
                    if l > 0:
                        resolved.append(l + 1)
                    else:
                        resolved.append(layer_idx + 1 + l)
                self.route_info[layer_idx] = resolved
                
            elif block['type'] == 'shortcut':
                from_idx = int(block['from'])
                resolved = layer_idx + 1 + from_idx
                self.shortcut_info[layer_idx] = resolved
            
            layer_idx += 1
    
    def _make_conv(self, block, in_ch, out_ch):
        kernel = int(block['size'])
        stride = int(block['stride'])
        pad = int(block.get('pad', 0))
        if pad:
            pad = kernel // 2
        groups = int(block.get('groups', 1))
        activation = block.get('activation', 'leaky')
        bn = int(block.get('batch_normalize', 0))
        
        if bn:
            return ConvBN(in_ch, out_ch, kernel, stride, pad, groups, activation)
        else:
            return ConvNoBN(in_ch, out_ch, kernel, stride, pad, groups, activation)
    
    def forward(self, x):
        outputs = [x]
        yolo_outputs = []
        
        layer_idx = 0
        for block in self.blocks:
            if block['type'] == 'net':
                continue
            
            module = self.module_list[layer_idx]
            
            if block['type'] == 'convolutional':
                x = module(outputs[-1])
                
            elif block['type'] == 'shortcut':
                from_idx = self.shortcut_info[layer_idx]
                x = outputs[-1] + outputs[from_idx]
                
            elif block['type'] == 'route':
                indices = self.route_info[layer_idx]
                if len(indices) == 1:
                    x = outputs[indices[0]]
                else:
                    x = torch.cat([outputs[i] for i in indices], dim=1)
                    
            elif block['type'] == 'maxpool':
                x = module(outputs[-1])
                
            elif block['type'] == 'upsample':
                x = module(outputs[-1])
                
            elif block['type'] == 'yolo':
                yolo_outputs.append(outputs[-1])
                x = outputs[-1]
                
            elif block['type'] == 'dropout':
                x = outputs[-1]
                
            else:
                x = outputs[-1]
            
            outputs.append(x)
            layer_idx += 1
        
        return yolo_outputs
    
    def load_darknet_weights(self, weights_path):
        """Load weights from Darknet .weights file."""
        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        
        print(f"Weights file contains {len(weights)} values")
        
        ptr = 0
        layer_idx = 0
        
        for block in self.blocks:
            if block['type'] == 'net':
                continue
            
            if block['type'] == 'convolutional':
                module = self.module_list[layer_idx]
                bn = int(block.get('batch_normalize', 0))
                
                if bn:
                    conv = module.conv
                    bn_layer = module.bn
                    
                    num_bn = bn_layer.bias.numel()
                    
                    bn_layer.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]))
                    ptr += num_bn
                    
                    bn_layer.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]))
                    ptr += num_bn
                    
                    bn_layer.running_mean.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]))
                    ptr += num_bn
                    
                    bn_layer.running_var.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]))
                    ptr += num_bn
                    
                    num_w = conv.weight.numel()
                    conv.weight.data.copy_(
                        torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight)
                    )
                    ptr += num_w
                else:
                    conv = module.conv
                    
                    num_b = conv.bias.numel()
                    conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b]))
                    ptr += num_b
                    
                    num_w = conv.weight.numel()
                    conv.weight.data.copy_(
                        torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight)
                    )
                    ptr += num_w
            
            layer_idx += 1
        
        print(f"Loaded {ptr} / {len(weights)} weights")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLO-Fastest to PyTorch')
    parser.add_argument('cfg', help='Path to .cfg file')
    parser.add_argument('weights', help='Path to .weights file')
    parser.add_argument('--output', '-o', default='yolo_fastest.pth', help='Output .pth file')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    
    args = parser.parse_args()
    
    model = YoloFastest(args.cfg, debug=args.debug)
    model.load_darknet_weights(args.weights)
    
    torch.save(model.state_dict(), args.output)
    print(f"Saved PyTorch weights to {args.output}")
