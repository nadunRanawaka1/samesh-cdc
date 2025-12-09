import os
import json
from collections import defaultdict
import numpy as np
import trimesh
import argparse
import copy
import open3d as o3d

from PIL import Image, ImageDraw, ImageFont
from pygltflib import GLTF2


from omegaconf import OmegaConf
from samesh.data.loaders import *
from samesh.models.sam_mesh import  colormap_faces_mesh, generate_high_contrast_palette, segment_mesh

from samesh.renderer.renderer import Renderer
from samesh.renderer.renderer import render_multiview
from samesh.utils.mesh import vertex_to_face_colors
from samesh.utils.vlm import Gemini, check_segmentation_prompt

def renumbered_face2label(face2label: dict):
    face2label = {int(k): int(v) for k, v in face2label.items()}
    label2face = defaultdict(list)
    for face, label in face2label.items():
        label2face[label].append(face)
    labels = sorted(list(label2face.keys()))
    renumbered_labels = {j: i for i, j in enumerate(labels, start=1)}  # Start from 1 to avoid label 0
    renumbered_face2label = {k: renumbered_labels[v] for k, v in face2label.items()}
    return renumbered_face2label

def mark_segments_with_numbers(
    image: Image.Image,
    face_ids: np.ndarray,
    face2label: dict[int, int],
    font_size: int = 40,
    outline_width: int = 3,
    min_segment_pixels: int = 100,
    min_distance: int = 80
) -> Image:
    """
    Mark each segment in the rendered image with its label number.
    Each number is colored with its segment's color and repositioned if too close to others.
    Automatically uses white outline for dark colors and black outline for light colors.
    
    Args:
        image: PIL Image of the rendered mesh
        face_ids: Array of face IDs per pixel from renderer (h x w)
        face2label: Dictionary mapping face IDs to segment labels
        font_size: Size of the text font
        outline_width: Width of text outline
        min_segment_pixels: Minimum number of pixels for a segment to get a label
        min_distance: Minimum distance between label centroids (will adjust if closer)
        
    Returns:
        PIL Image with segment numbers drawn on it
    """
    # Create a copy of the image to draw on
    labeled_image = image.copy()
    draw = ImageDraw.Draw(labeled_image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Generate the same high-contrast color palette used in colormap_faces_mesh
    label_max = max(face2label.values()) if face2label else 1
    palette = generate_high_contrast_palette(label_max + 1, seed=0)
    
    def get_luminance(color):
        """Calculate relative luminance using the formula for perceived brightness"""
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        return 0.299 * r + 0.587 * g + 0.114 * b
    
    def choose_outline_color(segment_color):
        """Choose white outline for dark colors, black for light colors"""
        luminance = get_luminance(segment_color)
        # If luminance is less than 0.5, the color is dark, use white outline
        return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)
    
    # Group pixels by segment label
    label2pixels = {}
    label0_count = 0
    
    for y in range(face_ids.shape[0]):
        for x in range(face_ids.shape[1]):
            face_id = face_ids[y, x]
            
            # Skip background pixels
            if face_id == -1:
                continue
                
            # Get the label for this face
            label = face2label.get(face_id, None)
            
            if label == 0:
                label0_count += 1
            
            if label is not None and label != 0:  # Skip background label 0
                if label not in label2pixels:
                    label2pixels[label] = []
                label2pixels[label].append((x, y))
    
    if label0_count > 0:
        print(f"Note: {label0_count} pixels have label 0 (treated as background, not labeled)")
    
    # Calculate centroids and colors for each segment
    label_info = []
    for label, pixels in label2pixels.items():
        # Skip if too few pixels
        if len(pixels) < min_segment_pixels:
            print(f"Skipping label {label}: only {len(pixels)} pixels (minimum {min_segment_pixels})")
            continue
            
        # Calculate centroid
        pixels_array = np.array(pixels)
        centroid_x = int(np.mean(pixels_array[:, 0]))
        centroid_y = int(np.mean(pixels_array[:, 1]))
        
        # Get segment color from palette
        segment_color = tuple(palette[label].tolist())
        outline_color = choose_outline_color(segment_color)
        
        label_info.append({
            'label': label,
            'x': centroid_x,
            'y': centroid_y,
            'color': segment_color,
            'outline_color': outline_color,
            'pixels': pixels_array
        })
    
    # Adjust positions to avoid overlaps
    adjusted_positions = []
    for i, info in enumerate(label_info):
        x, y = info['x'], info['y']
        
        # Check distance to all previously placed labels
        for prev_x, prev_y in adjusted_positions:
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            
            if distance < min_distance and distance > 0:
                # Move this label away from the previous one
                angle = np.arctan2(y - prev_y, x - prev_x)
                x = int(prev_x + min_distance * np.cos(angle))
                y = int(prev_y + min_distance * np.sin(angle))
                
                # Clamp to image bounds
                x = max(0, min(x, face_ids.shape[1] - 1))
                y = max(0, min(y, face_ids.shape[0] - 1))
        
        adjusted_positions.append((x, y))
        info['adjusted_x'] = x
        info['adjusted_y'] = y
    
    # Draw all labels
    for info in label_info:
        text = str(info['label'])
        x, y = info['adjusted_x'], info['adjusted_y']
        
        # Get text bounding box
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        text_x = x - text_width // 2
        text_y = y - text_height // 2
        
        # Draw outline with adaptive color (white for dark segments, black for light)
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx * dx + dy * dy <= outline_width * outline_width:
                    draw.text((text_x + dx, text_y + dy), text, font=font, fill=info['outline_color'])
        
        # Draw main text with segment color
        draw.text((text_x, text_y), text, font=font, fill=info['color'])
    
    return labeled_image

def render_mesh(mesh, output_dir, renderer_args, image_label=None, face2label=None):
    """
    Render a mesh with optional segmentation labeling
    
    Args:
        lighting_alpha: Diffuse lighting strength (0.0-1.0). Higher = stronger directional lighting
        lighting_beta: Ambient lighting level (0.0-1.0). Higher = brighter overall
    """
    
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh).to_mesh()
    mesh = vertex_to_face_colors(mesh)
   
    
    if face2label is not None:
        mesh_colored = colormap_faces_mesh(mesh, face2label, high_contrast=True)
    else:
        mesh_colored = mesh
    renderer = Renderer(renderer_args.copy())
    renderer.set_object(mesh_colored)
    renderer.set_camera()
    
    renders = render_multiview(renderer, camera_generation_method=renderer_args.camera_generation_method, renderer_args=renderer_args.renderer_args, sampling_args=renderer_args.sampling_args, lighting_args=renderer_args.lighting_args)

    for i, image in enumerate(renders['matte']):
        if face2label is not None:
            labeled_image = mark_segments_with_numbers(
                image, 
                renders['faces'][i], 
                face2label,
                font_size=40,
                min_segment_pixels=50,
                min_distance=80  # Minimum distance between nearby labels
            )
            labeled_image.save(f'{output_dir}/{image_label}_{i}_labeled.png')
        else:
            image.save(f'{output_dir}/{image_label}_{i}.png')
    return renders

def o3d_visualize_mesh(mesh_input: str):
    mesh = trimesh.load(mesh_input).to_mesh()
    # Convert to Open3D format
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    # Visualize
    o3d.visualization.draw_geometries([o3d_mesh])

def label_parts_and_render(config: OmegaConf, mesh_input: str, output_dir: str):

    config.output = Path(config.output) / Path(mesh_input).stem

    # After segmentation, label the parts
    labeled_dir = os.path.join(config.output,  'labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    face2label_file = f'{config.output}/face2label.json'
    with open(face2label_file, 'r') as f:
        face2label = json.load(f)
    face2label = renumbered_face2label(face2label)

    original_mesh = trimesh.load(mesh_input).to_mesh()
    segmented_mesh = trimesh.load(f'{config.output}/{Path(mesh_input).stem}_segmented.glb').to_mesh()
    
    # Prepare for rendering
    renderer_args = config.renderer.copy()
    renderer_args['sampling_args']['radius'] = 2.5
    renderer_args['renderer_args']['uv_map'] = True

    # Render the original mesh
    original_dir = os.path.join(config.output, 'original')
    os.makedirs(original_dir, exist_ok=True)
    original_renders = render_mesh(original_mesh, original_dir, renderer_args, image_label='original')


    #Render the segmented mesh
    renderer_args['sampling_args']['radius'] = 2.5
    segmented_renders = render_mesh(segmented_mesh, labeled_dir, renderer_args, image_label='segmented', face2label=face2label)
    return original_renders, segmented_renders
def main(config: str, mesh_input: str, output_dir: str):

    config = OmegaConf.load(config)
    if output_dir is not None:
        config.output = output_dir

    mesh = segment_mesh(mesh_input, config, visualize=True, target_labels=config.target_labels)

    original_renders, segmented_renders = label_parts_and_render(config, mesh_input, output_dir)

    # # Run VLM
    if "vlm" in config:
        gemini = Gemini(project=config.vlm.gemini.project, model=config.vlm.gemini.model)
        prompt = check_segmentation_prompt()
        prompt_images = []
        for i, image in enumerate(original_renders['matte']):
            prompt_images.append(image)
            prompt_images.append(segmented_renders['matte'][i])
        result = gemini(prompt, images=prompt_images)
        result_text = gemini.get_result_text(result)
        print(result_text)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mesh_segmentation_cdc.yaml')
    parser.add_argument('--mesh-input', type=str, default=None, required=True, help='Path to the input mesh .glb file.')
    parser.add_argument('--output-dir', type=str, default=None, required=False, help='Path to the output directory.')
    args = parser.parse_args()
    main(args.config, args.mesh_input, args.output_dir)