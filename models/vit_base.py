import timm
import torch.nn as nn

def get_vit_with_ape(img_size=32, patch_size=4, embed_dim=384, depth=12, 
                     num_heads=6, num_classes=10, drop_rate=0.1):
    """Returns a ViT with standard Absolute Positional Encoding"""
    
    model = timm.create_model(
        'vit_small_patch16_224',
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_rate=drop_rate,
        pretrained=False
    )
    
    # Adjust patch embedding for CIFAR-10 size if needed
    if img_size != 224:
        model.patch_embed.proj = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    return model