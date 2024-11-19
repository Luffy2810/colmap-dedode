import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import gc
from DeDoDe import dedode_detector_L, dedode_descriptor_G
from DeDoDe.utils import sample_keypoints
from ultralytics import YOLO
from utils import *
def to_pixel_coords(
    keypoints: Union[np.ndarray, torch.Tensor], 
    image_shape: Tuple[int, int]
) -> Union[np.ndarray, torch.Tensor]:
    """Convert keypoints from normalized coordinates [-1, 1] to pixel coordinates"""
    was_tensor = isinstance(keypoints, torch.Tensor)
    if was_tensor:
        keypoints = keypoints.cpu().numpy()
        
    keypoints = (keypoints + 1) / 2
    H, W = image_shape
    scaled_keypoints = keypoints * np.array([[W, H]])
    
    if was_tensor:
        scaled_keypoints = torch.from_numpy(scaled_keypoints)
        
    return scaled_keypoints

class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images"""
    
    def __init__(self, image_paths: List[Path], resize_h: int, resize_w: int):
        self.image_paths = image_paths
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pil_im = PIL.Image.open(image_path)
        original_size = pil_im.size  # (W, H)
        pil_im = pil_im.resize((self.resize_w, self.resize_h))
        image = np.array(pil_im) / 255.
        image = self.normalizer(torch.from_numpy(image).permute(2, 0, 1)).float()
        
        return {
            'original_image' : np.array(pil_im),
            'image': image,
            'path': str(image_path),
            'original_size': np.array(original_size)  # (W, H)
        }

class DeDoDeExtractor:
    """DeDoDe feature extractor with efficient batch processing"""
    
    def __init__(
        self,
        detector_weights: str,
        descriptor_weights: str,
        max_keypoints: int = 20000,
        remove_borders: bool = True,
        resize_h: int = 784,
        resize_w: int = 784*2,
        num_workers: int = 8,
        batch_size: int = 1 
    ):
        self.detector_weights = detector_weights
        self.descriptor_weights = descriptor_weights
        self.max_keypoints = max_keypoints
        self.remove_borders = remove_borders
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.detector = dedode_detector_L(
            weights=torch.load(self.detector_weights, map_location=self.device))
        self.descriptor = dedode_descriptor_G(
            weights=torch.load(self.descriptor_weights, map_location=self.device))
        
        self.detector.eval()
        self.descriptor.eval()
        self.yolo_model =  YOLO("/home/luffy/continue/ultralytics/runs/detect/train2/weights/best.pt")
        self.map_x_32, self.map_y_32 = generate_mapping_data(resize_w)

    @torch.no_grad()
    def extract_features_batch(self, image_batch: torch.Tensor, original_sizes: np.ndarray, original_images: np.ndarray) -> List[Dict]:
        """Extract features from a batch of images"""
        images = image_batch.to(self.device)
        B, C, H, W = images.shape

        # Detect keypoints
        logits = self.detector({"image": images})["keypoint_logits"]
        B, K, H, W = logits.shape
        keypoint_p = logits.reshape(B, K*H*W).softmax(dim=-1).reshape(B, K, H*W).sum(dim=1)

        # Sample keypoints
        keypoints_norm, scores = sample_keypoints(
            keypoint_p.reshape(B, H, W),
            use_nms=False,
            sample_topk=True,
            num_samples=self.max_keypoints,
            return_scoremap=True,
            sharpen=False,
            upsample=False,
            increase_coverage=True,
            remove_borders=self.remove_borders
        )
        
        # Get descriptors for the full batch first
        desc_grid = self.descriptor({"image": images})["description_grid"]
        descriptors = F.grid_sample(
            desc_grid.float(),
            keypoints_norm[:, None],
            mode="bilinear",
            align_corners=False
        )[:, :, 0].mT
        
        # Lists to store filtered data
        filtered_features = []
        
        for i, (image, keypoints) in enumerate(zip(original_images, keypoints_norm)):
            keypoints_net = to_pixel_coords(keypoints, (H, W))
            scale_w = original_sizes[i][0] / W
            scale_h = original_sizes[i][1] / H
            scale_factors = np.array([scale_w, scale_h])
            keypoints_orig = keypoints_net.cpu().numpy() * scale_factors[None]
            keypoints_cpu = keypoints_orig

            cubemap = cv2.remap(image, self.map_x_32, self.map_y_32, cv2.INTER_LANCZOS4)
            keypoints_cube, faces_0 = vectorized_equirectangular_to_cubemap(keypoints_cpu, self.resize_w, self.resize_h, self.resize_w)

            preds = self.yolo_model(cubemap, verbose=False)
            bboxes = np.array([])
            try:
                bboxes = np.array([
                    [int(boxes.xyxy[0][0].cpu()), int(boxes.xyxy[0][1].cpu()), 
                    int(boxes.xyxy[0][2].cpu()), int(boxes.xyxy[0][3].cpu())]
                    for result in preds
                    for boxes in [result.boxes]
                ])
            except:
                pass

            # Initialize mask for keeping points
            keep_mask = torch.ones(len(keypoints), dtype=torch.bool, device=keypoints.device)
            
            if len(bboxes) > 0:
                keypoints_1 = keypoints_cube
                x_within = (keypoints_1[:, 0:1] >= bboxes[:, 0]) & (keypoints_1[:, 0:1] <= bboxes[:, 2])
                y_within = (keypoints_1[:, 1:2] >= bboxes[:, 1]) & (keypoints_1[:, 1:2] <= bboxes[:, 3])
                inside_bbox = np.any(x_within & y_within, axis=1)
                keep_mask = torch.tensor(~inside_bbox, dtype=torch.bool, device=keypoints.device)

            # Filter the data using the mask
            curr_keypoints = keypoints[keep_mask]
            curr_keypoints_orig = keypoints_orig[keep_mask.cpu().numpy()]
            curr_scores = scores[i][keep_mask]
            curr_descriptors = descriptors[i][keep_mask]

            features = {
                "keypoints": curr_keypoints_orig,
                "keypoints_normalized": curr_keypoints.cpu().numpy(),
                "scores": curr_scores.cpu().numpy(),
                "descriptors": curr_descriptors.cpu().numpy(),
                "image_size": original_sizes[i]
            }
            filtered_features.append(features)

        return filtered_features

    def process_images(self, image_dir: str, output_path: str, 
                      image_patterns: List[str] = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"],
                      as_half: bool = True) -> None:
        """Process multiple images using batched processing"""
        # Get image paths
        image_paths = []
        image_dir = Path(image_dir)
        for pattern in image_patterns:
            image_paths.extend(image_dir.glob(f"**/{pattern}"))
        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, self.resize_h, self.resize_w)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

        # Process images
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_path), "a") as f:
            for batch in tqdm(dataloader, desc="Processing images"):
                # try:
                    # Extract features for the batch
                batch_features = self.extract_features_batch(
                    batch['image'], 
                    batch['original_size'].numpy(),
                    batch['original_image'].numpy()
                )
                
                # Save features for each image in the batch
                for features, image_path in zip(batch_features, batch['path']):
                    # Convert to float16 if requested
                    if as_half:
                        for k, v in features.items():
                            if isinstance(v, np.ndarray) and v.dtype == np.float32:
                                features[k] = v.astype(np.float16)
                    
                    # Save to H5 file
                    name = Path(image_path).name
                    if name in f:
                        del f[name]
                        
                    grp = f.create_group(name)
                    for k, v in features.items():
                        grp.create_dataset(k, data=v)
                
                # except Exception as e:
                #     print(f"Error processing batch: {str(e)}")
                #     for path in batch['path']:
                #         print(f"Affected image: {path}")
                del batch_features
                gc.collect()
                torch.cuda.empty_cache()  

    def extract_single(self, image_path: str) -> Dict:
        """Extract features from a single image"""
        dataset = ImageDataset([Path(image_path)], self.resize_h, self.resize_w)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        batch = next(iter(dataloader))
        features = self.extract_features_batch(batch['image'], batch['original_size'].numpy(),batch['original_image'].numpy())
        return features[0]


if __name__ == "__main__":
    extractor = DeDoDeExtractor(
        detector_weights="weights/dedode_detector_L_v2.pth",
        descriptor_weights="weights/dedode_descriptor_G.pth",
        num_workers=4  # Adjust based on your CPU
    )

    # Process a directory of images
    extractor.process_images(
        image_dir="/home/luffy/data/66bb7830903efc9c1f67c941_frames_1568_784",
        output_path="output/features.h5",
        as_half=True
    )

    # Or process a single image
    # features = extractor.extract_single("path/to/image.jpg")