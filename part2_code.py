import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rays(height, width, intrinsics, w_R_c, w_T_c):
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    #############################  TODO 2.1 BEGIN  ##########################
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))
    x, y = x.float(), y.float()
    #normalization
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(height*width, 3).T
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords)
    ray_directions = torch.matmul(w_R_c,cam_coords).T
    ray_directions = ray_directions.reshape(height, width, 3)  # (H,W,3)
    ray_origins = torch.reshape(w_T_c, (1, 1, 3)).expand(height, width, 3)
    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions


def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """
    #############################  TODO 2.2 BEGIN  ############################
    H, W = ray_directions.shape[0], ray_directions.shape[1]    
    N = samples
    ray_points = torch.zeros((H, W, N, 3))
    depth_points = torch.zeros((H, W, N))

    # sampled points and depth of each pixel sample
    for i in range(N):
        t_i = near + ((i - 1) / N) * (far - near)
        ray_points[:, :, i, :] = ray_origins + t_i * ray_directions             # for every pixel sample
        depth_points[:, :, i] = torch.full_like(depth_points[:, :, i], t_i)     # depth value for each pixel sample in the image

    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points

class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        # for autograder compliance, please follow the given naming for your layers
        gamma_x = 3 * (1 + 2 * num_x_frequencies)
        gamma_d = 3 * (1 + 2 * num_d_frequencies)
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(gamma_x, filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),
            'layer_6': nn.Linear(filter_size + gamma_x, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_s': nn.Linear(filter_size, 1),
            'layer_9': nn.Linear(filter_size, filter_size),
            'layer_10': nn.Linear(filter_size + gamma_d, 128),
            'layer_11': nn.Linear(128, 3),
        })
        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        # example of forward through a layer: y = self.layers['layer_1'](x)
        x1 = F.relu(self.layers['layer_1'](x))
        x2 = F.relu(self.layers['layer_2'](x1))
        x3 = F.relu(self.layers['layer_3'](x2))
        x4 = F.relu(self.layers['layer_4'](x3))
        x5 = F.relu(self.layers['layer_5'](x4))
        x5_prime = torch.cat([x5, x], dim=-1)

        x6 = F.relu(self.layers['layer_6'](x5_prime))
        x7 = F.relu(self.layers['layer_7'](x6))
        x8 = F.relu(self.layers['layer_8'](x7))
        
        sigma = self.layers['layer_s'](x8) #x9

        x10 = self.layers['layer_9'](x8) # no actiavtion, normal layer
        x10_prime = torch.cat([x10, d], dim=-1)
        
        x11 = F.relu(self.layers['layer_10'](x10_prime))
        rgb = torch.sigmoid(self.layers['layer_11'](x11)) #sigmoid activation
        
        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
    def positional_encoding(x, num_frequencies=6, incl_input=True):
        """
        Apply positional encoding to the input.
        """
        results = []
        if incl_input:
            results.append(x)
        for L in range(num_frequencies):
            results.append(torch.sin(2 ** L * torch.pi * x))
            results.append(torch.cos(2 ** L * torch.pi * x))
        return torch.cat(results, dim=-1)
    # Normalize the ray directions
    ray_directions_norm = F.normalize(ray_directions, p=2, dim=2).to(device)

    # Populate ray directions along each ray
    ray_directions_expanded = torch.unsqueeze(ray_directions_norm, 2)
    ray_directions_expanded = ray_directions_expanded.expand(-1, -1, ray_points.size(2), -1)
    # flatten
    ray_directions = ray_directions_expanded.reshape(-1, 3)
    # positional encoding
    ray_directions_encoded = positional_encoding(ray_directions, num_d_frequencies)
    
    # flatten
    ray_points = ray_points.reshape(-1, 3)
    # positional encoding
    ray_points_encoded = positional_encoding(ray_points, num_x_frequencies)

    # Chunk encoded ray points and directions
    ray_points_batches = get_chunks(ray_points_encoded)
    ray_directions_batches = get_chunks(ray_directions_encoded)

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    #############################  TODO 2.4 BEGIN  ############################
    
    del_n = 1e9
    
    delta_depth = torch.full_like(depth_points, del_n).to(rgb.device)
    delta_depth[..., :-1] = depth_points[..., 1:] - depth_points[..., :-1]

    sigma_del = F.relu(s) * delta_depth.reshape_as(s)
    e_sigma = torch.exp(-sigma_del)
    T = torch.cumprod(e_sigma, dim=-1)
    T = torch.roll(T, shifts=1, dims=-1)

    C_r = (T * (1 - e_sigma))[..., None] * rgb
    rec_image = torch.sum(C_r, dim=-2)

    #############################  TODO 2.4 END  ############################

    return rec_image


def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):

    #############################  TODO 2.5 BEGIN  ############################

    #compute all the rays from the image
    w_R_c = pose[:3,:3]
    w_T_c = pose[:3,3]
    ray_origins, ray_directions = get_rays(height, width, intrinsics, w_R_c, w_T_c)

    #sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_points_batches, ray_directions_batches = get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)


    #forward pass the batches and concatenate the outputs at the end
    rgb_list, density_list = [], []
    for i in range(len(ray_directions_batches)):
        rgb, density = model.forward(ray_points_batches[i], ray_directions_batches[i])
        rgb_list.append(rgb)
        density_list.append(density)
    
    rgb_list = torch.cat(rgb_list)
    rgb_list = rgb_list.reshape(height, width, samples, 3) #(H,W,N,3)
    
    density_list = torch.cat(density_list)
    density_list = density_list.reshape(height, width, samples) #(H,W,N)

    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb_list, density_list, depth_points).to(device)

    #############################  TODO 2.5 END  ############################

    return rec_image