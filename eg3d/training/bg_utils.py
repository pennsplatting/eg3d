import math
import torch
import torch.nn.functional as F

def hemispherical_texture_map(direction, texture):
    """
    Maps a normalized vector to a hemispherical texture and returns the corresponding RGB color.

    :param direction: A tuple or list of three elements representing the normalized look-at direction. [N, 3]
    :param texture: [3, H, W]
    :return: A tuple containing the RGB color. [N, 3]
    """
    # Load the texture image
    C, texture_width, texture_height = texture.shape

    # Convert the normalized vector into spherical coordinates
    x, y, z = direction.split([1,1,1], dim=1)
    r = (x**2 + y**2 + z**2)**0.5
    
    # if r == 0:
    #     theta = 0
    #     phi = 0
    # else:
    #     theta = math.acos(z / r)
    #     phi = math.atan2(y, x)
    zero = torch.zeros_like(r)
    theta = torch.where(
        r == 0,
        zero,
        torch.acos(z / r)
    )
    
    phi = torch.where(
        r == 0,
        zero,
        torch.atan2(y, x)
    )

    # Ensure that we are in the correct hemisphere (-90° to 90° for the 'phi' angle).
    
    # if not (-math.pi <= phi).all() and (phi <= math.pi).all():
    #     print(phi.max(), phi.min())
    #     raise ValueError("The look-at direction is outside the specified hemisphere range.")

    # Map spherical coordinates to 2D texture coordinates
    # Note that u is scaled to [0, 1] over half the circle since we're only considering one hemisphere.
    u = phi / (2 * math.pi)
    v = theta / math.pi

    # Map 2D texture coordinates to image pixel coordinates
    tex_x = torch.tensor(u * texture_width, dtype=torch.float) % texture_width
    tex_y = torch.tensor((1 - v) * texture_height, dtype=torch.float) % texture_height
    coords = torch.cat((tex_x,tex_y),dim=-1).unsqueeze(dim=1)
    print(coords)

    # Query the RGB value from the texture
    rgb_color = F.grid_sample(texture[None], coords[None], align_corners=False)
    rgb_color = rgb_color.squeeze().reshape(C, texture_width, texture_height)
    return rgb_color

# Example usage:
# direction = (0, 0, 1)  # Replace with your own normalized vector within the range -90° to 90°
# texture_path = 'path_to_your_texture.jpg'  # Replace with the path to your texture image
# try:
#     color = hemispherical_texture_map(direction, texture_path)
#     print(color)
# except ValueError as e:
#     print(e)
