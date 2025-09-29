"""
Utility functions for working with Surya layout JSON format.
Handles conversion between different Surya output formats.
"""

import json
from typing import Dict, List


def parse_surya_raw_output(raw_output_str: str) -> Dict:
    """
    Parse raw Surya output string (like from your example) into usable dict.
    
    The raw output looks like:
    [LayoutResult(bboxes=[LayoutBox(polygon=..., bbox=...), ...], ...)]
    
    This function extracts the structured data from it.
    
    Args:
        raw_output_str: String representation of Surya output
        
    Returns:
        Dict with 'bboxes' and 'image_bbox'
    """
    # Note: This is a simplified parser. In practice, you'd use the actual
    # LayoutResult object directly from Surya, not parse strings.
    
    # For the example you provided, we can manually extract the key data
    # In real usage, you'd have the LayoutResult object directly
    
    raise NotImplementedError(
        "String parsing not implemented. Use the LayoutResult object directly "
        "or save/load from JSON instead."
    )


def save_layout_to_json(layout_result, output_path: str):
    """
    Save a Surya LayoutResult object to JSON file.
    
    Args:
        layout_result: Surya LayoutResult object
        output_path: Path to save JSON file
    """
    boxes = []
    for box in layout_result.bboxes:
        boxes.append({
            'bbox': box.bbox,
            'polygon': box.polygon,
            'label': box.label,
            'confidence': box.confidence,
            'position': box.position,
            'top_k': box.top_k if hasattr(box, 'top_k') else None
        })
    
    layout_dict = {
        'bboxes': boxes,
        'image_bbox': layout_result.image_bbox,
        'sliced': layout_result.sliced if hasattr(layout_result, 'sliced') else False
    }
    
    with open(output_path, 'w') as f:
        json.dump(layout_dict, f, indent=2)
    
    print(f"✓ Layout saved to: {output_path}")


def load_layout_from_json(json_path: str) -> Dict:
    """
    Load a saved layout JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dict with layout data
    """
    with open(json_path, 'r') as f:
        layout_dict = json.load(f)
    
    # Ensure it has required fields
    assert 'bboxes' in layout_dict, "JSON must contain 'bboxes'"
    assert 'image_bbox' in layout_dict, "JSON must contain 'image_bbox'"
    
    return layout_dict


def filter_boxes_by_label(layout_dict: Dict, labels: List[str]) -> List[Dict]:
    """
    Filter layout boxes by label(s).
    
    Args:
        layout_dict: Layout dictionary
        labels: List of label strings to keep (e.g., ['Figure', 'Picture'])
        
    Returns:
        List of filtered boxes
    """
    filtered = []
    for box in layout_dict['bboxes']:
        if box['label'] in labels:
            filtered.append(box)
    
    return filtered


def get_boxes_in_region(
    layout_dict: Dict, 
    region_bbox: List[float],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Get all layout boxes that overlap with a specific region.
    
    Args:
        layout_dict: Layout dictionary
        region_bbox: [x1, y1, x2, y2] region of interest
        iou_threshold: Minimum IoU to consider (0.0 to 1.0)
        
    Returns:
        List of boxes that overlap the region
    """
    def compute_iou(box1, box2):
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    overlapping = []
    for box in layout_dict['bboxes']:
        iou = compute_iou(box['bbox'], region_bbox)
        if iou >= iou_threshold:
            box_with_iou = box.copy()
            box_with_iou['iou'] = iou
            overlapping.append(box_with_iou)
    
    # Sort by IoU (highest first)
    overlapping.sort(key=lambda b: -b['iou'])
    
    return overlapping


def visualize_layout_boxes(
    image_path: str,
    layout_dict: Dict,
    output_path: str = None,
    labels_to_show: List[str] = None
):
    """
    Draw bounding boxes on the image for visualization.
    
    Args:
        image_path: Path to the page image
        layout_dict: Layout dictionary
        output_path: Where to save visualization (if None, displays)
        labels_to_show: Only show these labels (if None, shows all)
    """
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Color map for different labels
    colors = {
        'Figure': 'red',
        'Picture': 'red',
        'Code': 'blue',
        'Text': 'green',
        'SectionHeader': 'purple',
        'Caption': 'orange',
        'Table': 'cyan',
        'PageHeader': 'gray',
        'PageFooter': 'gray',
        'Footnote': 'brown'
    }
    
    for box in layout_dict['bboxes']:
        label = box['label']
        
        # Skip if not in filter list
        if labels_to_show and label not in labels_to_show:
            continue
        
        bbox = box['bbox']
        x1, y1, x2, y2 = bbox
        
        color = colors.get(label, 'yellow')
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label text
        text = f"{label} ({box['confidence']:.2f})"
        draw.text((x1, y1 - 15), text, fill=color)
    
    if output_path:
        img.save(output_path)
        print(f"✓ Visualization saved to: {output_path}")
    else:
        img.show()


def analyze_layout_statistics(layout_dict: Dict) -> Dict:
    """
    Analyze layout and return statistics.
    
    Args:
        layout_dict: Layout dictionary
        
    Returns:
        Dict with statistics
    """
    from collections import Counter
    
    # Count labels
    label_counts = Counter(box['label'] for box in layout_dict['bboxes'])
    
    # Calculate average confidence per label
    label_confidences = {}
    for label in label_counts.keys():
        confidences = [
            box['confidence'] 
            for box in layout_dict['bboxes'] 
            if box['label'] == label
        ]
        label_confidences[label] = sum(confidences) / len(confidences)
    
    # Get image dimensions
    img_bbox = layout_dict['image_bbox']
    img_width = img_bbox[2] - img_bbox[0]
    img_height = img_bbox[3] - img_bbox[1]
    
    # Calculate coverage (% of image covered by boxes)
    total_area = 0
    for box in layout_dict['bboxes']:
        bbox = box['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        total_area += width * height
    
    coverage_pct = (total_area / (img_width * img_height)) * 100
    
    stats = {
        'total_boxes': len(layout_dict['bboxes']),
        'label_counts': dict(label_counts),
        'label_confidences': label_confidences,
        'image_dimensions': [img_width, img_height],
        'coverage_percentage': coverage_pct
    }
    
    return stats


def print_layout_summary(layout_dict: Dict):
    """
    Print a human-readable summary of the layout.
    
    Args:
        layout_dict: Layout dictionary
    """
    stats = analyze_layout_statistics(layout_dict)
    
    print("\n" + "="*60)
    print("LAYOUT SUMMARY")
    print("="*60)
    print(f"Total boxes: {stats['total_boxes']}")
    print(f"Image size: {stats['image_dimensions'][0]:.0f} x {stats['image_dimensions'][1]:.0f} px")
    print(f"Coverage: {stats['coverage_percentage']:.1f}%")
    
    print("\nLabel Distribution:")
    for label, count in sorted(stats['label_counts'].items(), key=lambda x: -x[1]):
        conf = stats['label_confidences'][label]
        print(f"  {label:20s}: {count:3d} boxes (avg conf: {conf:.3f})")
    
    print("="*60 + "\n")


def create_minimal_layout_dict(boxes_data: List[Dict], image_bbox: List[float] = None) -> Dict:
    """
    Create a minimal layout dict for testing.
    
    Args:
        boxes_data: List of box dicts with at least 'bbox' and 'label'
        image_bbox: Optional image bbox (default: [0, 0, 3000, 4000])
        
    Returns:
        Minimal layout dict
    """
    if image_bbox is None:
        image_bbox = [0.0, 0.0, 3000.0, 4000.0]
    
    # Ensure all boxes have required fields
    processed_boxes = []
    for i, box_data in enumerate(boxes_data):
        box = {
            'bbox': box_data['bbox'],
            'label': box_data['label'],
            'confidence': box_data.get('confidence', 0.95),
            'position': box_data.get('position', i)
        }
        processed_boxes.append(box)
    
    return {
        'bboxes': processed_boxes,
        'image_bbox': image_bbox
    }


# ========== Example Usage ==========

if __name__ == "__main__":
    print("Surya JSON Utilities - Examples\n")
    
    # Example 1: Create a test layout from your document
    print("="*60)
    print("Example 1: Creating Layout from Your Document Example")
    print("="*60)
    
    # This matches the data structure from your document
    test_layout = {
        'bboxes': [
            {
                'bbox': [68.4814453125, 871.728515625, 140.6982421875, 2312.255859375],
                'label': 'PageHeader',
                'confidence': 0.9996969699859619,
                'position': 0
            },
            {
                'bbox': [494.3115234375, 435.05859375, 2050.7080078125, 499.51171875],
                'label': 'SectionHeader',
                'confidence': 0.9997674822807312,
                'position': 1
            },
            {
                'bbox': [486.8408203125, 596.19140625, 2048.2177734375, 770.21484375],
                'label': 'Text',
                'confidence': 0.9961956739425659,
                'position': 2
            },
            {
                'bbox': [1329.78515625, 955.517578125, 2276.07421875, 1738.623046875],
                'label': 'Picture',
                'confidence': 0.4757022559642792,
                'position': 11
            },
            {
                'bbox': [1319.82421875, 1825.634765625, 2305.95703125, 2183.349609375],
                'label': 'Caption',
                'confidence': 0.9927271604537964,
                'position': 12
            }
        ],
        'image_bbox': [0.0, 0.0, 2550.0, 3300.0]
    }
    
    # Print summary
    print_layout_summary(test_layout)
    
    # Example 2: Filter boxes by label
    print("Example 2: Filter Boxes by Label")
    print("-"*60)
    
    figures = filter_boxes_by_label(test_layout, ['Figure', 'Picture'])
    print(f"Found {len(figures)} figure/picture boxes:")
    for fig in figures:
        print(f"  - {fig['label']} at {fig['bbox']} (conf: {fig['confidence']:.3f})")
    
    # Example 3: Find boxes in a region
    print("\n" + "="*60)
    print("Example 3: Find Boxes in Region")
    print("-"*60)
    
    # Define a region of interest (e.g., right column)
    roi = [1200.0, 900.0, 2400.0, 2200.0]
    print(f"Region of interest: {roi}")
    
    boxes_in_roi = get_boxes_in_region(test_layout, roi, iou_threshold=0.3)
    print(f"\nFound {len(boxes_in_roi)} boxes overlapping the region:")
    for box in boxes_in_roi:
        print(f"  - {box['label']:15s} IoU: {box['iou']:.3f}")
    
    # Example 4: Save and load layout
    print("\n" + "="*60)
    print("Example 4: Save and Load Layout JSON")
    print("-"*60)
    
    output_json = "test_layout.json"
    
    # Save
    with open(output_json, 'w') as f:
        json.dump(test_layout, f, indent=2)
    print(f"✓ Saved layout to: {output_json}")
    
    # Load
    loaded_layout = load_layout_from_json(output_json)
    print(f"✓ Loaded layout with {len(loaded_layout['bboxes'])} boxes")
    
    # Example 5: Create minimal layout for testing
    print("\n" + "="*60)
    print("Example 5: Create Minimal Layout for Testing")
    print("-"*60)
    
    test_boxes = [
        {
            'bbox': [100, 200, 500, 600],
            'label': 'Figure'
        },
        {
            'bbox': [100, 650, 500, 720],
            'label': 'Caption'
        }
    ]
    
    minimal_layout = create_minimal_layout_dict(test_boxes)
    print(f"✓ Created minimal layout with {len(minimal_layout['bboxes'])} boxes")
    print(f"  Image bbox: {minimal_layout['image_bbox']}")
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)