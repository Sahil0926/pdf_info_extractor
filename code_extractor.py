"""
Code/Algorithm Extractor Script
Extracts code blocks and algorithms from PDFs using Surya layout analysis.
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


class CodeExtractor:
    """Extracts code/algorithm blocks from PDF pages using semantic search and layout analysis."""
    
    def __init__(self, dpi: int = 144):
        """
        Initialize the code extractor.
        
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
            query_text: Text to search for (e.g., "Algorithm 1", "pseudocode")
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
    
    def _boxes_intersect(self, bbox1: List[float], bbox2: List[float], threshold: float = 0.0) -> bool:
        """
        Check if two bounding boxes intersect.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            threshold: IoU threshold (0.0 = any overlap, 1.0 = complete overlap)
            
        Returns:
            True if boxes intersect
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Check if boxes overlap
        if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
            return False
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold
    
    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """
        Merge two bounding boxes into one (union).
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            Merged bbox [x1, y1, x2, y2]
        """
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return [x1, y1, x2, y2]
    
    def _merge_intersecting_code_boxes(self, candidates: List[Dict]) -> List[Dict]:
        """
        Merge intersecting code boxes into single boxes.
        
        Args:
            candidates: List of code candidate dicts
            
        Returns:
            List of merged candidates
        """
        if len(candidates) <= 1:
            return candidates
        
        # Create a copy to work with
        merged = []
        used = set()
        
        for i, cand1 in enumerate(candidates):
            if i in used:
                continue
            
            # Start with this candidate
            merged_bbox = cand1['bbox'].copy()
            merged_confidence = cand1['confidence']
            merged_positions = [cand1.get('position', i)]
            merge_count = 1
            
            # Check all other candidates
            for j, cand2 in enumerate(candidates[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if they intersect
                if self._boxes_intersect(merged_bbox, cand2['bbox']):
                    # Merge them
                    merged_bbox = self._merge_bboxes(merged_bbox, cand2['bbox'])
                    # Take max confidence
                    merged_confidence = max(merged_confidence, cand2['confidence'])
                    merged_positions.append(cand2.get('position', j))
                    used.add(j)
                    merge_count += 1
            
            # Add merged candidate
            merged_candidate = {
                'label': cand1['label'],
                'bbox': merged_bbox,
                'confidence': merged_confidence,
                'candidate_type': cand1.get('candidate_type', 'primary_code'),
                'position': merged_positions[0],  # Keep first position
                'merged_from': merged_positions if merge_count > 1 else None,
                'merge_count': merge_count
            }
            
            merged.append(merged_candidate)
            used.add(i)
        
        return merged
    
    def filter_code_candidates(
        self, 
        layout_dict: Dict,
        query_text: str = ""
    ) -> List[Dict]:
        """
        Filter layout boxes to find code/algorithm candidates.
        Merges intersecting code boxes.
        
        Primary: 'Code' label
        Fallback: Text blocks near SectionHeaders containing "Algorithm", "Pseudo", "Code"
        
        Args:
            layout_dict: Layout JSON dict
            query_text: Original search query (for fallback heuristics)
            
        Returns:
            List of candidate dicts with 'label', 'bbox', and 'candidate_type'
        """
        candidates = []
        boxes = layout_dict['bboxes']
        
        # Primary: Look for 'Code' labeled blocks
        for box in boxes:
            if box['label'] == 'Code':
                candidates.append({
                    'label': box['label'],
                    'bbox': box['bbox'],  # [x1, y1, x2, y2]
                    'confidence': box['confidence'],
                    'candidate_type': 'primary_code',
                    'position': box['position']
                })
        
        # Merge intersecting code boxes
        if candidates:
            candidates = self._merge_intersecting_code_boxes(candidates)
            print("Candidates merged")
            return candidates
        
        # Fallback: Look for algorithm-related SectionHeaders
        algorithm_keywords = ['algorithm', 'pseudo', 'code', 'listing']
        
        # Find SectionHeaders that might indicate algorithms
        algo_headers = []
        for i, box in enumerate(boxes):
            if box['label'] == 'SectionHeader':
                # Check if header text likely contains algorithm reference
                # (We don't have OCR here, so we use position-based heuristics)
                algo_headers.append({
                    'bbox': box['bbox'],
                    'position': i
                })
        
        # If we have algorithm headers, look for Text blocks nearby
        if algo_headers:
            for header in algo_headers:
                hdr_x1, hdr_y1, hdr_x2, hdr_y2 = header['bbox']
                hdr_center_y = (hdr_y1 + hdr_y2) / 2
                
                # Look for Text blocks below this header
                for box in boxes:
                    if box['label'] == 'Text' and box['position'] > header['position']:
                        txt_x1, txt_y1, txt_x2, txt_y2 = box['bbox']
                        
                        # Check if text is below header and reasonably close
                        if txt_y1 > hdr_y2 and (txt_y1 - hdr_y2) < 200:
                            # Calculate height (taller blocks more likely to be code)
                            height = txt_y2 - txt_y1
                            
                            candidates.append({
                                'label': 'Text',
                                'bbox': box['bbox'],
                                'confidence': box['confidence'],
                                'candidate_type': 'fallback_text_near_header',
                                'position': box['position'],
                                'height': height,
                                'distance_from_header': txt_y1 - hdr_y2
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
    
    def point_in_bbox(self, point: Tuple[float, float], bbox: List[float]) -> bool:
        """Check if a point is inside a bounding box."""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def select_best_code(
        self, 
        candidates: List[Dict], 
        text_bbox: List[float]
    ) -> Tuple[Optional[Dict], List[str]]:
        """
        Select the best code block candidate relative to matched text (header/reference).
        
        Heuristic:
        1. Prefer code blocks that CONTAIN the matched text (text inside code)
        2. Else prefer code blocks BELOW the matched text (header above code)
        3. Tie-break by largest height (code blocks are usually tall)
        
        Args:
            candidates: List of code candidate dicts
            text_bbox: Matched text bbox [x1, y1, x2, y2] in image coords
            
        Returns:
            Tuple of (selected_candidate or None, notes list)
        """
        notes = []
        
        if not candidates:
            return None, ["No code/algorithm candidates found"]
        
        if len(candidates) == 1:
            notes.append("Only one candidate found, selected by default")
            return candidates[0], notes
        
        # Text reference bbox
        txt_x1, txt_y1, txt_x2, txt_y2 = text_bbox
        txt_center = ((txt_x1 + txt_x2) / 2, (txt_y1 + txt_y2) / 2)
        
        # Categorize candidates
        containing_candidates = []
        below_candidates = []
        other_candidates = []
        
        for cand in candidates:
            code_x1, code_y1, code_x2, code_y2 = cand['bbox']
            
            # Check if code block contains the text center point
            if self.point_in_bbox(txt_center, cand['bbox']):
                cand['height'] = code_y2 - code_y1
                containing_candidates.append(cand)
            # Check if code block is below the text
            elif code_y1 > txt_y2:
                cand['vertical_distance'] = code_y1 - txt_y2
                cand['height'] = code_y2 - code_y1
                below_candidates.append(cand)
            else:
                # Code is above or overlapping in complex ways
                code_center_y = (code_y1 + code_y2) / 2
                cand['vertical_distance'] = abs(code_center_y - txt_center[1])
                cand['height'] = code_y2 - code_y1
                other_candidates.append(cand)
        
        # Priority 1: Code blocks containing the text
        if containing_candidates:
            # Sort by largest height (bigger code blocks preferred)
            containing_candidates.sort(key=lambda c: -c['height'])
            selected = containing_candidates[0]
            notes.append(f"Selected code block containing the matched text (height={selected['height']:.1f}px)")
            if len(containing_candidates) > 1:
                notes.append(f"Other {len(containing_candidates)-1} containing blocks were discarded")
            return selected, notes
        
        # Priority 2: Code blocks below the text (typical: header above code)
        if below_candidates:
            # Sort by: min vertical distance, then max height
            below_candidates.sort(
                key=lambda c: (c['vertical_distance'], -c['height'])
            )
            selected = below_candidates[0]
            notes.append(f"Selected code block below text (distance={selected['vertical_distance']:.1f}px, height={selected['height']:.1f}px)")
            if len(below_candidates) > 1:
                notes.append(f"Other {len(below_candidates)-1} blocks below were discarded")
            return selected, notes
        
        # Priority 3: Fallback to nearest code block
        if other_candidates:
            other_candidates.sort(
                key=lambda c: (c['vertical_distance'], -c['height'])
            )
            selected = other_candidates[0]
            notes.append(f"Selected nearest code block (distance={selected['vertical_distance']:.1f}px)")
            return selected, notes
        
        return None, ["No suitable code blocks found relative to matched text"]
    
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
        
        output_path = os.path.join(output_dir, f"final_code_page_{page_number:03d}.png")
        cropped.save(output_path)
        
        return output_path
    
    # ==================== MAIN EXTRACTION METHOD ====================
    
    def extract_code(
        self,
        pdf_path: str,
        query_text: str,
        output_dir: str,
        page_hint: Optional[int] = None,
        layout_json: Optional[Dict] = None
    ) -> Dict:
        """
        Main method to extract a code/algorithm block from PDF.
        
        Args:
            pdf_path: Path to PDF file
            query_text: Text to search for (e.g., "Algorithm 1", "pseudocode")
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
        candidates = self.filter_code_candidates(layout_dict, query_text)
        
        # Add merge information to notes
        merge_notes = []
        for cand in candidates:
            if cand.get('merged_from') and cand.get('merge_count', 1) > 1:
                positions = cand['merged_from']
                merge_notes.append(
                    f"Merged {cand['merge_count']} intersecting code boxes "
                    f"(positions: {positions}) into one box with confidence {cand['confidence']:.3f}"
                )
        
        # Step 5: Select best code block
        selected, notes = self.select_best_code(candidates, text_bbox_img)
        
        # Prepend merge notes
        notes = merge_notes + notes
        
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
    extractor = CodeExtractor(dpi=144)
    
    result = extractor.extract_code(
        pdf_path="./paper.pdf",
        query_text="Algorithm 1",
        output_dir="./output/algo_/",
        page_hint=None,  # Will search all pages
        layout_json=None  # Will run Surya
    )
    
    print(f"Page: {result['page']}")
    print(f"Matched text: {result['matched_text']}")
    print(f"Selected region: {result['selected_region']}")
    print(f"Cropped image: {result['cropped_image_path']}")
    print(f"Notes: {result['notes']}")