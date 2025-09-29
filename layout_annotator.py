"""
Layout Annotator Utility
Annotates Surya layout boxes on images with customizable styles.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


class LayoutAnnotator:
    """Annotate layout boxes on images for visualization and debugging."""
    
    # Default color scheme for different labels
    DEFAULT_COLORS = {
        'Figure': '#FF0000',      # Red
        'Picture': '#FF0000',     # Red
        'Code': '#0066FF',        # Blue
        'Text': '#00CC00',        # Green
        'SectionHeader': '#9933FF',  # Purple
        'Caption': '#FF6600',     # Orange
        'Table': '#00CCCC',       # Cyan
        'PageHeader': '#808080',  # Gray
        'PageFooter': '#808080',  # Gray
        'Footnote': '#996633',    # Brown
        'List': '#FF66CC',        # Pink
        'Form': '#FFCC00',        # Yellow
    }
    
    def __init__(
        self, 
        box_width: int = 3,
        font_size: int = 16,
        show_confidence: bool = True,
        show_position: bool = False,
        show_bbox_coords: bool = False,
        custom_colors: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the annotator.
        
        Args:
            box_width: Width of bounding box lines (default: 3)
            font_size: Font size for labels (default: 16)
            show_confidence: Show confidence scores (default: True)
            show_position: Show position numbers (default: False)
            show_bbox_coords: Show bbox coordinates (default: False)
            custom_colors: Custom color map {label: hex_color}
        """
        self.box_width = box_width
        self.font_size = font_size
        self.show_confidence = show_confidence
        self.show_position = show_position
        self.show_bbox_coords = show_bbox_coords
        
        # Merge custom colors with defaults
        self.colors = self.DEFAULT_COLORS.copy()
        if custom_colors:
            self.colors.update(custom_colors)
        
        # Try to load a better font
        self.font = self._load_font()
    
    def _load_font(self):
        """Try to load a nice font, fallback to default."""
        try:
            # Try to load a TrueType font
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_size)
        except:
            try:
                return ImageFont.truetype("arial.ttf", self.font_size)
            except:
                # Fallback to default font
                return ImageFont.load_default()
    
    def _get_color(self, label: str) -> str:
        """Get color for a label, with fallback."""
        return self.colors.get(label, '#FFFF00')  # Yellow as fallback
    
    def _format_label_text(self, box: Dict) -> str:
        """Format the text label for a box."""
        parts = [box['label']]
        
        if self.show_confidence:
            conf = box.get('confidence', 0)
            parts.append(f"{conf:.2f}")
        
        if self.show_position:
            pos = box.get('position', -1)
            parts.append(f"#{pos}")
        
        if self.show_bbox_coords:
            bbox = box['bbox']
            parts.append(f"[{bbox[0]:.0f},{bbox[1]:.0f}]")
        
        return " | ".join(parts)
    
    def annotate_image(
        self,
        image_path: str,
        layout_json_path: str,
        output_path: str,
        labels_to_show: Optional[List[str]] = None,
        highlight_boxes: Optional[List[int]] = None,
        draw_background: bool = True
    ) -> str:
        """
        Annotate an image with layout boxes.
        
        Args:
            image_path: Path to the input image
            layout_json_path: Path to the layout JSON file
            output_path: Path to save annotated image
            labels_to_show: Only show these labels (None = show all)
            highlight_boxes: List of box positions to highlight with thicker borders
            draw_background: Draw semi-transparent background behind text labels
            
        Returns:
            Path to the saved annotated image
        """
        # Load image
        img = Image.open(image_path).convert('RGBA')
        
        # Create a transparent overlay for semi-transparent elements
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Create main drawing context
        draw = ImageDraw.Draw(img)
        
        # Load layout JSON
        with open(layout_json_path, 'r') as f:
            layout_dict = json.load(f)
        
        # Sort boxes by position for consistent drawing order
        boxes = sorted(layout_dict['bboxes'], key=lambda b: b.get('position', 0))
        
        # Draw all boxes
        for box in boxes:
            label = box['label']
            
            # Filter by label if specified
            if labels_to_show and label not in labels_to_show:
                continue
            
            bbox = box['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get color
            color = self._get_color(label)
            
            # Determine box width (highlight if in highlight list)
            position = box.get('position', -1)
            width = self.box_width * 2 if (highlight_boxes and position in highlight_boxes) else self.box_width
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            # Format label text
            text = self._format_label_text(box)
            
            # Calculate text position and size
            try:
                # For newer Pillow versions
                bbox_text = draw.textbbox((x1, y1 - self.font_size - 4), text, font=self.font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                # Fallback for older versions
                text_width, text_height = draw.textsize(text, font=self.font)
            
            # Position text above the box
            text_x = x1
            text_y = y1 - text_height - 8
            
            # Adjust if text goes off top of image
            if text_y < 0:
                text_y = y1 + 4
            
            # Draw background rectangle for text
            if draw_background:
                padding = 4
                bg_bbox = [
                    text_x - padding,
                    text_y - padding,
                    text_x + text_width + padding,
                    text_y + text_height + padding
                ]
                
                # Convert hex color to RGB with alpha
                rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                bg_color = rgb + (200,)  # 200/255 opacity
                
                draw_overlay.rectangle(bg_bbox, fill=bg_color)
            
            # Draw text
            draw.text((text_x, text_y), text, fill=color, font=self.font)
        
        # Composite the overlay onto the main image
        img = Image.alpha_composite(img, overlay)
        
        # Convert back to RGB for saving
        img = img.convert('RGB')
        
        # Save
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path, quality=95)
        
        return output_path
    
    def annotate_from_dict(
        self,
        image_path: str,
        layout_dict: Dict,
        output_path: str,
        labels_to_show: Optional[List[str]] = None,
        highlight_boxes: Optional[List[int]] = None,
        draw_background: bool = True
    ) -> str:
        """
        Annotate an image using a layout dictionary directly.
        
        Args:
            image_path: Path to the input image
            layout_dict: Layout dictionary (not a file path)
            output_path: Path to save annotated image
            labels_to_show: Only show these labels (None = show all)
            highlight_boxes: List of box positions to highlight
            draw_background: Draw semi-transparent background behind text
            
        Returns:
            Path to the saved annotated image
        """
        # Save layout dict to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(layout_dict, f)
            temp_json_path = f.name
        
        try:
            result = self.annotate_image(
                image_path=image_path,
                layout_json_path=temp_json_path,
                output_path=output_path,
                labels_to_show=labels_to_show,
                highlight_boxes=highlight_boxes,
                draw_background=draw_background
            )
        finally:
            # Clean up temp file
            os.unlink(temp_json_path)
        
        return result
    
    def create_comparison_view(
        self,
        image_path: str,
        layout_json_path: str,
        output_path: str,
        labels_to_compare: List[str]
    ) -> str:
        """
        Create a side-by-side comparison of different label types.
        
        Args:
            image_path: Path to input image
            layout_json_path: Path to layout JSON
            output_path: Path to save comparison image
            labels_to_compare: List of label types to show separately
            
        Returns:
            Path to saved comparison image
        """
        # Load image and layout
        base_img = Image.open(image_path)
        
        with open(layout_json_path, 'r') as f:
            layout_dict = json.load(f)
        
        # Create a grid of images
        num_views = len(labels_to_compare) + 1  # +1 for "all" view
        
        # Calculate grid layout
        cols = min(3, num_views)
        rows = (num_views + cols - 1) // cols
        
        # Create composite image
        img_width, img_height = base_img.size
        composite_width = img_width * cols
        composite_height = img_height * rows
        composite = Image.new('RGB', (composite_width, composite_height), 'white')
        
        # Draw context
        draw = ImageDraw.Draw(composite)
        
        # Create individual annotated views
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # View 0: All labels
            all_view_path = os.path.join(tmpdir, 'all.png')
            self.annotate_image(image_path, layout_json_path, all_view_path)
            all_img = Image.open(all_view_path)
            
            x, y = 0, 0
            composite.paste(all_img, (x, y))
            draw.text((x + 10, y + 10), "ALL LABELS", fill='black', font=self.font)
            
            # Individual label views
            for idx, label in enumerate(labels_to_compare, start=1):
                view_path = os.path.join(tmpdir, f'{label}.png')
                self.annotate_image(
                    image_path, 
                    layout_json_path, 
                    view_path,
                    labels_to_show=[label]
                )
                label_img = Image.open(view_path)
                
                # Calculate position in grid
                col = idx % cols
                row = idx // cols
                x = col * img_width
                y = row * img_height
                
                composite.paste(label_img, (x, y))
                draw.text((x + 10, y + 10), label.upper(), fill='black', font=self.font)
        
        # Save composite
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        composite.save(output_path, quality=95)
        
        return output_path
    
    def create_legend(self, output_path: str, width: int = 400, height: int = 600) -> str:
        """
        Create a legend image showing all label colors.
        
        Args:
            output_path: Path to save legend image
            width: Width of legend image
            height: Height of legend image
            
        Returns:
            Path to saved legend image
        """
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Title
        title = "Layout Label Colors"
        draw.text((20, 20), title, fill='black', font=self.font)
        
        # Draw color swatches
        y_offset = 60
        box_size = 30
        spacing = 40
        
        for label, color in sorted(self.colors.items()):
            # Draw color box
            x1, y1 = 20, y_offset
            x2, y2 = x1 + box_size, y1 + box_size
            
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            draw.rectangle([x1, y1, x2, y2], fill=rgb, outline='black', width=2)
            
            # Draw label text
            draw.text((x1 + box_size + 15, y1 + 5), label, fill='black', font=self.font)
            
            y_offset += spacing
            
            # Check if we need to wrap to a second column
            if y_offset > height - spacing:
                break
        
        # Save
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)
        
        return output_path


# ==================== Convenience Functions ====================

def quick_annotate(
    image_path: str,
    json_path: str,
    output_folder: str,
    output_filename: Optional[str] = None
) -> str:
    """
    Quick annotation with default settings.
    
    Args:
        image_path: Path to image
        json_path: Path to layout JSON
        output_folder: Output folder
        output_filename: Optional custom filename (default: annotated_<original_name>)
        
    Returns:
        Path to annotated image
    """
    # Create output filename
    if output_filename is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"annotated_{name}{ext}"
    
    output_path = os.path.join(output_folder, output_filename)
    
    # Annotate
    annotator = LayoutAnnotator()
    return annotator.annotate_image(image_path, json_path, output_path)


def annotate_with_highlights(
    image_path: str,
    json_path: str,
    output_folder: str,
    highlight_labels: List[str],
    output_filename: Optional[str] = None
) -> str:
    """
    Annotate with specific labels highlighted.
    
    Args:
        image_path: Path to image
        json_path: Path to layout JSON
        output_folder: Output folder
        highlight_labels: Labels to show (others will be hidden)
        output_filename: Optional custom filename
        
    Returns:
        Path to annotated image
    """
    if output_filename is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        labels_str = "_".join(highlight_labels)
        output_filename = f"annotated_{name}_{labels_str}{ext}"
    
    output_path = os.path.join(output_folder, output_filename)
    
    annotator = LayoutAnnotator(box_width=4)
    return annotator.annotate_image(
        image_path, 
        json_path, 
        output_path,
        labels_to_show=highlight_labels
    )


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("Layout Annotator - Examples\n")
    
    # Example 1: Basic annotation
    print("="*60)
    print("Example 1: Basic Annotation")
    print("="*60)
    
    annotator = LayoutAnnotator(
        box_width=3,
        font_size=16,
        show_confidence=True,
        show_position=False
    )
    
    result = annotator.annotate_image(
        image_path="./output/figures/page_000.png",
        layout_json_path="./output/figures/layout_page_000.json",
        output_path="./output/figures/annotated_img.png"
    )
    print(f"✓ Annotated image saved to: {result}\n")
    
    # # Example 2: Annotate only specific labels
    # print("="*60)
    # print("Example 2: Highlight Only Figures and Code")
    # print("="*60)
    
    # result = annotator.annotate_image(
    #     image_path="page_000.png",
    #     layout_json_path="layout_page_000.json",
    #     output_path="./output/figures_and_code.png",
    #     labels_to_show=['Figure', 'Picture', 'Code']
    # )
    # print(f"✓ Filtered annotation saved to: {result}\n")
    
    # # Example 3: Highlight specific boxes
    # print("="*60)
    # print("Example 3: Highlight Specific Boxes")
    # print("="*60)
    
    # result = annotator.annotate_image(
    #     image_path="page_000.png",
    #     layout_json_path="layout_page_000.json",
    #     output_path="./output/highlighted_boxes.png",
    #     highlight_boxes=[11, 12]  # Positions to highlight
    # )
    # print(f"✓ Highlighted annotation saved to: {result}\n")
    
    # # Example 4: Custom colors
    # print("="*60)
    # print("Example 4: Custom Color Scheme")
    # print("="*60)
    
    # custom_annotator = LayoutAnnotator(
    #     custom_colors={
    #         'Figure': '#FF1493',  # Deep pink
    #         'Code': '#00FF00',    # Bright green
    #         'Text': '#4169E1'     # Royal blue
    #     }
    # )
    
    # result = custom_annotator.annotate_image(
    #     image_path="page_000.png",
    #     layout_json_path="layout_page_000.json",
    #     output_path="./output/custom_colors.png"
    # )
    # print(f"✓ Custom color annotation saved to: {result}\n")
    
    # # Example 5: Create comparison view
    # print("="*60)
    # print("Example 5: Comparison View")
    # print("="*60)
    
    # result = annotator.create_comparison_view(
    #     image_path="./output/algo_/page_010.png",
    #     layout_json_path="./output/algo_/layout_page_010.json",
    #     output_path="./output/figures/",
    #     labels_to_compare=None
    # )
    # print(f"✓ Comparison view saved to: {result}\n")
    
    # # Example 6: Create color legend
    # print("="*60)
    # print("Example 6: Create Color Legend")
    # print("="*60)
    
    # result = annotator.create_legend("./output/legend.png")
    # print(f"✓ Legend saved to: {result}\n")
    
    # # Example 7: Quick annotation
    # print("="*60)
    # print("Example 7: Quick Annotation (Convenience Function)")
    # print("="*60)
    
    # result = quick_annotate(
    #     image_path="page_000.png",
    #     json_path="layout_page_000.json",
    #     output_folder="./output"
    # )
    # print(f"✓ Quick annotation saved to: {result}\n")
    
    # # Example 8: Annotate from dict (no JSON file)
    # print("="*60)
    # print("Example 8: Annotate from Dictionary")
    # print("="*60)
    
    # test_layout = {
    #     'bboxes': [
    #         {
    #             'bbox': [100, 200, 500, 600],
    #             'label': 'Figure',
    #             'confidence': 0.95,
    #             'position': 0
    #         },
    #         {
    #             'bbox': [100, 650, 500, 720],
    #             'label': 'Caption',
    #             'confidence': 0.98,
    #             'position': 1
    #         }
    #     ],
    #     'image_bbox': [0, 0, 2550, 3300]
    # }
    
    # result = annotator.annotate_from_dict(
    #     image_path="page_000.png",
    #     layout_dict=test_layout,
    #     output_path="./output/from_dict.png"
    # )
    # print(f"✓ Annotation from dict saved to: {result}\n")
    
    # print("="*60)
    # print("All examples complete!")
    # print("="*60)
