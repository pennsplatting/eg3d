import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
# import torch_utils.persistence as persistence
from PIL import Image

# @persistence.persistent_class
class EdgeExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.w = torch.tensor(np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]).reshape(1, 1, 3, 3), dtype=torch.float)
        # self.w = torch.tensor(np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).reshape(1, 1, 3, 3), dtype=torch.float)
        # self.w = torch.tensor(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).reshape(1, 1, 3, 3), dtype=torch.float)
        # self.w = torch.tensor(np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).reshape(1, 1, 3, 3), dtype=torch.float)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device="cuda").view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, device="cuda").view(1, 1, 3, 3)
        # self.conv = nn.Conv2d(1, 1, (3, 3), padding=1)
        # self.conv.weight = nn.Parameter(self.w)
        self.transform = transforms.Grayscale(num_output_channels=1)
        
    def forward(self, img):
        img = self.transform(img)
        
        grad_x = torch.nn.functional.conv2d(img, self.sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(img, self.sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad
        # return self.transform(img)
    
if __name__ == '__main__':
    edgeextractor = EdgeExtractor()
    real_img = np.array(Image.open('/home/zxy/eg3d/eg3d/data/ffhq/FFHQ_512/00000/img00000239.png'))
    real_img = torch.tensor(real_img, dtype=torch.float, device="cuda")
    print(real_img.shape)
    real_img = real_img.permute(2,0,1).unsqueeze(0)
    print(real_img.shape)
    with torch.no_grad():
        real_img_edge = edgeextractor(real_img).squeeze().detach().cpu().numpy()
        Image.fromarray(real_img_edge.astype('uint8')).save('/home/zxy/eg3d/img00000239_grayscale.png')

        
        