"""
Figure/Image Extractor Script
Extracts figures and images from PDFs using Surya layout analysis.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import fitz  # PyMuPDF
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings


class FigureExtractor:
    """Extracts figures/images from PDF pages using semantic search and layout analysis."""
    
    def __init__(self, dpi: int = 144):
        """
        Initialize the figure extractor.
        
        Args:
            dpi: DPI for page rasterization (default: 144)
        """
        self.dpi = dpi
        self.zoom = dpi / 72.0  # PDF is 72 DPI by default
        self.layout_predictor = None
        
    def _init_layout_predictor(self):
        """Lazy initialization of Surya layout predictor."""
        if self.layout_predictor is None:
            foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
            self.layout_predictor = LayoutPredictor(foundation)
    
    # ==================== STEP 1: Semantic Text Search ====================
    
    def search_text_in_pdf(
        self, 
        pdf_path: str, 
        query_text: str, 
        page_hint: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Search for query text in PDF and return the best match with location.
        
        Args:
            pdf_path: Path to PDF file
            query_text: Text to search for (e.g., "Figure 2", "Fig. 3")
            page_hint: Optional page number to search (0-indexed)
            
        Returns:
            Dict with: page_number, matched_text, pdf_bbox [x0, y0, x1, y1]
            None if no match found
        """
        doc = fitz.open(pdf_path)
        best_match = None
        best_score = 0
        
        pages_to_search = [page_hint] if page_hint is not None else range(len(doc))
        
        for page_num in pages_to_search:
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            # Search for text (case-insensitive)
            text_instances = page.search_for(query_text, quads=False)
            
            if text_instances:
                # Take the first instance on this page
                rect = text_instances[0]
                match_info = {
                    'page_number': page_num,
                    'matched_text': query_text,
                    'pdf_bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                    'score': 1.0  # Exact match
                }
                
                if match_info['score'] > best_score:
                    best_match = match_info
                    best_score = match_info['score']
        
        doc.close()
        return best_match
    
    # ==================== STEP 2: Page Rasterization ====================
    
    def rasterize_page(
        self, 
        pdf_path: str, 
        page_number: int, 
        output_dir: str
    ) -> Tuple[str, Image.Image]:
        """
        Convert a single PDF page to an image.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-indexed)
            output_dir: Directory to save the image
            
        Returns:
            Tuple of (image_path, PIL.Image)
        """
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        
        # Create transformation matrix for zoom
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"page_{page_number:03d}.png")
        pix.save(output_path)
        
        # Also load as PIL Image for Surya
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        return output_path, img
    
    # ==================== STEP 3: Layout JSON ====================
    
    def get_or_compute_layout(
        self,
        page_image: Image.Image,
        page_number: int,
        output_dir: str,
        layout_json: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        Get layout JSON either from provided data or by running Surya.
        
        Args:
            page_image: PIL Image of the page
            page_number: Page number (0-indexed)
            output_dir: Directory to save layout JSON
            layout_json: Optional pre-computed layout JSON
            
        Returns:
            Tuple of (json_path, layout_dict)
        """
        json_path = os.path.join(output_dir, f"layout_page_{page_number:03d}.json")
        
        if layout_json is not None:
            # Use provided layout
            layout_dict = layout_json
        else:
            # Run Surya layout prediction
            self._init_layout_predictor()
            layout_predictions = self.layout_predictor([page_image])
            layout_dict = self._convert_layout_to_dict(layout_predictions[0])
        
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(layout_dict, f, indent=2)
        
        return json_path, layout_dict
    
    def _convert_layout_to_dict(self, layout_result) -> Dict:
        """Convert Surya LayoutResult to serializable dict."""
        boxes = []
        for box in layout_result.bboxes:
            boxes.append({
                'bbox': box.bbox,  # [x1, y1, x2, y2]
                'label': box.label,
                'confidence': box.confidence,
                'position': box.position
            })
        
        return {
            'bboxes': boxes,
            'image_bbox': layout_result.image_bbox
        }
    
    # ==================== STEP 4: Candidate Region Filtering ====================
    
    def filter_figure_candidates(self, layout_dict: Dict) -> List[Dict]:
        """
        Filter layout boxes to find figure/picture candidates.
        
        Args:
            layout_dict: Layout JSON dict
            
        Returns:
            List of candidate dicts with 'label' and 'bbox'
        """
        candidates = []
        target_labels = {'Figure', 'Picture'}
        
        for box in layout_dict['bboxes']:
            if box['label'] in target_labels:
                candidates.append({
                    'label': box['label'],
                    'bbox': box['bbox'],  # [x1, y1, x2, y2]
                    'confidence': box['confidence']
                })
        
        return candidates
    
    # ==================== STEP 5: Region Selection (Spatial Heuristic) ====================
    
    def pdf_bbox_to_image_coords(
        self, 
        pdf_bbox: List[float], 
        pdf_height: float,
        padding: int = 6
    ) -> List[float]:
        """
        Convert PDF coordinates to image coordinates.
        
        Args:
            pdf_bbox: [x0, y0, x1, y1] in PDF coords (origin bottom-left)
            pdf_height: Height of PDF page in points
            padding: Padding to add in pixels
            
        Returns:
            [x1, y1, x2, y2] in image coords (origin top-left)
        """
        # PDF: origin at bottom-left, y increases upward
        # Image: origin at top-left, y increases downward
        x0, y0, x1, y1 = pdf_bbox
        
        # Convert to image coordinates
        img_x1 = x0 * self.zoom - padding
        img_y1 = (pdf_height - y1) * self.zoom - padding  # Flip y
        img_x2 = x1 * self.zoom + padding
        img_y2 = (pdf_height - y0) * self.zoom + padding  # Flip y
        
        return [img_x1, img_y1, img_x2, img_y2]
    
    def select_best_figure(
        self, 
        candidates: List[Dict], 
        text_bbox: List[float]
    ) -> Tuple[Optional[Dict], List[str]]:
        """
        Select the best figure candidate relative to matched text (caption).
        
        Heuristic: Prefer figures ABOVE the caption, with max vertical overlap.
        
        Args:
            candidates: List of figure candidate dicts
            text_bbox: Matched text bbox [x1, y1, x2, y2] in image coords
            
        Returns:
            Tuple of (selected_candidate or None, notes list)
        """
        notes = []
        
        if not candidates:
            return None, ["No figure/picture candidates found"]
        
        if len(candidates) == 1:
            notes.append("Only one candidate found, selected by default")
            return candidates[0], notes
        
        # Text caption bbox
        txt_x1, txt_y1, txt_x2, txt_y2 = text_bbox
        txt_center_x = (txt_x1 + txt_x2) / 2
        
        # Separate candidates above and below caption
        above_candidates = []
        below_candidates = []
        
        for cand in candidates:
            fig_x1, fig_y1, fig_x2, fig_y2 = cand['bbox']
            
            # Check if figure is above caption (figure's bottom < caption's top)
            if fig_y2 < txt_y1:
                # Calculate horizontal overlap
                overlap_x1 = max(fig_x1, txt_x1)
                overlap_x2 = min(fig_x2, txt_x2)
                overlap = max(0, overlap_x2 - overlap_x1)
                
                cand['vertical_distance'] = txt_y1 - fig_y2
                cand['horizontal_overlap'] = overlap
                cand['width'] = fig_x2 - fig_x1
                above_candidates.append(cand)
            else:
                fig_center_y = (fig_y1 + fig_y2) / 2
                cand['vertical_distance'] = abs(fig_center_y - txt_y1)
                below_candidates.append(cand)
        
        # Prefer figures above caption
        if above_candidates:
            # Sort by: max horizontal overlap, then min vertical distance, then max width
            above_candidates.sort(
                key=lambda c: (-c['horizontal_overlap'], c['vertical_distance'], -c['width'])
            )
            selected = above_candidates[0]
            notes.append(f"Selected figure above caption (overlap={selected['horizontal_overlap']:.1f}px)")
            if len(above_candidates) > 1:
                notes.append(f"Other {len(above_candidates)-1} figures above caption were discarded")
            return selected, notes
        
        # Fallback: choose nearest figure below caption
        if below_candidates:
            below_candidates.sort(key=lambda c: c['vertical_distance'])
            selected = below_candidates[0]
            notes.append(f"No figures above caption; selected nearest figure below (distance={selected['vertical_distance']:.1f}px)")
            return selected, notes
        
        return None, ["No suitable figures found relative to caption"]
    
    # ==================== STEP 6: Crop and Save ====================
    
    def crop_and_save(
        self, 
        page_image: Image.Image, 
        bbox: List[float], 
        page_number: int,
        output_dir: str
    ) -> str:
        """
        Crop region from page image and save.
        
        Args:
            page_image: PIL Image of the page
            bbox: [x1, y1, x2, y2] to crop
            page_number: Page number (0-indexed)
            output_dir: Directory to save crop
            
        Returns:
            Path to saved cropped image
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(page_image.width, x2)
        y2 = min(page_image.height, y2)
        
        cropped = page_image.crop((x1, y1, x2, y2))
        
        output_path = os.path.join(output_dir, f"final_figure_page_{page_number:03d}.png")
        cropped.save(output_path)
        
        return output_path
    
    # ==================== MAIN EXTRACTION METHOD ====================
    
    def extract_figure(
        self,
        pdf_path: str,
        query_text: str,
        output_dir: str,
        page_hint: Optional[int] = None,
        layout_json: Optional[Dict] = None
    ) -> Dict:
        """
        Main method to extract a figure from PDF.
        
        Args:
            pdf_path: Path to PDF file
            query_text: Text to search for (e.g., "Figure 2")
            output_dir: Output directory for all files
            page_hint: Optional page number hint (0-indexed)
            layout_json: Optional pre-computed layout JSON
            
        Returns:
            Result dict with all extraction details
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Search for text
        match = self.search_text_in_pdf(pdf_path, query_text, page_hint)
        if not match:
            return {
                'page': None,
                'matched_text': None,
                'matched_text_bbox': None,
                'candidates': [],
                'selected_region': None,
                'layout_json_path': None,
                'page_image_path': None,
                'cropped_image_path': None,
                'notes': ['No text match found']
            }
        
        page_num = match['page_number']
        pdf_bbox = match['pdf_bbox']
        print(match, "Checking match")
        
        # Step 2: Rasterize page
        page_image_path, page_image = self.rasterize_page(pdf_path, page_num, output_dir)
        
        # Step 3: Get or compute layout
        layout_json_path, layout_dict = self.get_or_compute_layout(
            page_image, page_num, output_dir, layout_json
        )
        
        # Convert PDF bbox to image coordinates
        doc = fitz.open(pdf_path)
        pdf_height = doc[page_num].rect.height
        doc.close()
        
        text_bbox_img = self.pdf_bbox_to_image_coords(pdf_bbox, pdf_height)
        
        # Step 4: Filter candidates
        candidates = self.filter_figure_candidates(layout_dict)
        
        # Step 5: Select best figure
        selected, notes = self.select_best_figure(candidates, text_bbox_img)
        
        # Step 6: Crop and save
        cropped_path = None
        if selected:
            cropped_path = self.crop_and_save(
                page_image, selected['bbox'], page_num, output_dir
            )
        
        return {
            'page': page_num,
            'matched_text': match['matched_text'],
            'matched_text_bbox': text_bbox_img,
            'candidates': candidates,
            'selected_region': selected,
            'layout_json_path': layout_json_path,
            'page_image_path': page_image_path,
            'cropped_image_path': cropped_path,
            'notes': notes
        }


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Example usage
    extractor = FigureExtractor(dpi=144)
    
    result = extractor.extract_figure(
        pdf_path="./paper.pdf",
        query_text="Figure 1.",
        output_dir="./output/figures/",
        page_hint=None,  # Will search all pages
        layout_json=None  # Will run Surya
    )
    
    print(f"Page: {result['page']}")
    print(f"Matched text: {result['matched_text']}")
    print(f"Selected region: {result['selected_region']}")
    print(f"Cropped image: {result['cropped_image_path']}")
    print(f"Notes: {result['notes']}")


