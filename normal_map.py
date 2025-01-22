from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_light_matrix(light_pth):
    extension = light_pth.split(".")[-1]
    if extension == "yml":
        fs = cv2.FileStorage(light_pth, cv2.FILE_STORAGE_READ)
        lights_node = fs.getNode("Lights")
        data = lights_node.mat()
        fs.release()        
        return np.array(data)
    
    elif extension == "txt":
        with open(light_pth, "r") as f:
            data = f.readlines()
        
        array = np.array([list(map(float, line.split())) for line in data])
        return array.T


def generate_normal_map(images_matrix, light_matrix):
    images_matrix = np.array(images_matrix)     # num_images x h x w
    light_matrix = np.array(light_matrix)       # num_images x 3
    transpose_light_matrix = np.transpose(light_matrix)     # 3 x num_images
    A  = np.dot(transpose_light_matrix, light_matrix)       # 3 x 3
    b = np.einsum('ij,jkl->ikl', transpose_light_matrix, images_matrix)  # 3 x h x w
    G  = np.einsum('ij,jkl->ikl', np.linalg.inv(A), b)      # 3 x h x w
    
    albedo  = np.linalg.norm(G, axis=0, keepdims=True)      # 1 x h x w
    albedo[albedo < 0] = 0
    normal_map = np.transpose(G, (1, 2, 0)).astype(int)     # h x w x 3
    normal_map[normal_map < 0] = 0
    return normal_map, albedo[0]

def main():
    
    light_pth = "samples/hippo/refined_light.txt"
    mask_pth = None
    image_pth = sorted(glob("samples/hippo/Objects/*.png"))
    
    # Example 1:
    # light_pth = "samples/ball/LightMatrix.yml"
    # mask_pth = "samples/ball/mask.bmp"
    # image_pth = [
    #     "samples/ball/0.bmp",
    #     "samples/ball/1.bmp",
    #     "samples/ball/2.bmp",
    #     "samples/ball/3.bmp",
    # ]
    
    # Example 2:
    # light_pth = "samples/buddha/LightMatrix.yml"
    # mask_pth = "samples/buddha/mask.png"
    # image_pth = [
    #     "samples/buddha/buddha0.png",
    #     "samples/buddha/buddha1.png",
    #     "samples/buddha/buddha2.png",
    #     "samples/buddha/buddha3.png",
    #     "samples/buddha/buddha4.png",
    #     "samples/buddha/buddha5.png",
    #     "samples/buddha/buddha6.png",
    #     "samples/buddha/buddha7.png",
    #     "samples/buddha/buddha8.png",
    #     "samples/buddha/buddha9.png",
    #     "samples/buddha/buddha10.png",
    #     "samples/buddha/buddha11.png",
    # ]

    # Example 3:
    # light_pth = "samples/shrek/LightMatrix.yml"
    # mask_pth = "samples/shrek/mask.bmp"
    # image_pth = [
    #     "samples/shrek/shrek0.bmp",
    #     "samples/shrek/shrek1.bmp",
    #     "samples/shrek/shrek2.bmp",
    #     "samples/shrek/shrek3.bmp",
    #     "samples/shrek/shrek4.bmp",
    #     "samples/shrek/shrek5.bmp",
    #     "samples/shrek/shrek6.bmp",
    #     "samples/shrek/shrek7.bmp",
    # ]

    images = []
    for pth in image_pth:
        images.append(
            cv2.imread(pth, flags=cv2.IMREAD_GRAYSCALE)
        )
    
    light_matrix = get_light_matrix(light_pth)
    normal_map, albedo = generate_normal_map(images, light_matrix)

    if mask_pth:
        mask = cv2.imread(mask_pth, flags=cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask[mask == 255] = 1
        mask = mask[:, :, np.newaxis]
        normal_map = normal_map * mask
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(normal_map)
    plt.title("Normal map")

    plt.subplot(1, 2, 2)
    plt.imshow(albedo, cmap="gray")
    plt.title("Albedo")
    plt.show()

if __name__ == "__main__":
    main()
