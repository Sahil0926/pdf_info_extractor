# PDF Figure & Code Extractor

Two separate Python scripts for extracting figures/images and code/algorithm blocks from PDFs using Surya layout analysis.

## 📁 Files

- **`figure_extractor.py`** - Extract figures and images from PDFs
- **`code_extractor.py`** - Extract code blocks and algorithms from PDFs
- **`extraction_demo.py`** - Examples showing how to use both extractors
- **`surya_json_utils.py`** - Utilities for working with Surya layout JSON
- **`layout_annotator.py`** - Annotating Surya layout JSON 

## 🎯 Features

Both scripts:
- ✅ Semantic text search in PDF to locate references (e.g., "Figure 3", "Algorithm 1")
- ✅ Convert only the selected page to an image
- ✅ Run or accept pre-computed Surya layout JSON
- ✅ Smart spatial heuristics to select the correct region
- ✅ Deterministic file naming
- ✅ Comprehensive result dictionaries with all details

## 📦 Dependencies

```bash
pip install PyMuPDF Pillow surya-ocr
```

## 🚀 Quick Start

### Extract a Figure

```python
from figure_extractor import FigureExtractor

extractor = FigureExtractor(dpi=144)

result = extractor.extract_figure(
    pdf_path="paper.pdf",
    query_text="Figure 2",
    output_dir="./output/figures",
    page_hint=None,  # Will search all pages
    layout_json=None  # Will run Surya automatically
)

print(f"Figure saved to: {result['cropped_image_path']}")
```

### Extract Code/Algorithm

```python
from code_extractor import CodeExtractor

extractor = CodeExtractor(dpi=144)

result = extractor.extract_code(
    pdf_path="paper.pdf",
    query_text="Algorithm 1",
    output_dir="./output/code",
    page_hint=3,  # Search only page 3
    layout_json=None
)

print(f"Code saved to: {result['cropped_image_path']}")
```

## 📋 Input Parameters

### Common to Both Scripts

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pdf_path` | `str` | Yes | Path to PDF file |
| `query_text` | `str` | Yes | Text to search for (e.g., "Figure 2", "Algorithm 1") |
| `output_dir` | `str` | Yes | Output directory for all generated files |
| `page_hint` | `int` or `None` | No | Page number (0-indexed) to search, or `None` for all pages |
| `dpi` | `int` | No | DPI for rasterization (default: 144) |
| `layout_json` | `dict` or `None` | No | Pre-computed Surya layout, or `None` to run Surya |

## 📤 Output Structure

Both scripts return a dictionary with:

```python
{
    'page': int,                        # Page number (0-indexed)
    'matched_text': str,                # Text that was matched
    'matched_text_bbox': [x1, y1, x2, y2],  # Location of matched text
    'candidates': [                     # All candidate regions found
        {'label': str, 'bbox': [...], 'confidence': float, ...}
    ],
    'selected_region': {                # The selected region (or None)
        'label': str,
        'bbox': [x1, y1, x2, y2],
        'confidence': float,
        ...
    },
    'layout_json_path': str,            # Path to saved layout JSON
    'page_image_path': str,             # Path to rasterized page image
    'cropped_image_path': str or None,  # Path to final cropped image
    'notes': [str]                      # Human-readable notes about decisions
}
```

## 📂 Output Files

### File Naming Convention

- **Page image**: `page_{page:03d}.png`
- **Layout JSON**: `layout_page_{page:03d}.json`
- **Final crop (figure)**: `final_figure_page_{page:03d}.png`
- **Final crop (code)**: `final_code_page_{page:03d}.png`

### Example Output Directory

```
output/
├── figures/
│   ├── page_000.png
│   ├── layout_page_000.json
│   └── final_figure_page_000.png
└── code/
    ├── page_003.png
    ├── layout_page_003.json
    └── final_code_page_003.png
```

## 🔍 How It Works

### Script A: Figure Extractor

**Labels of Interest**: `Figure`, `Picture`

**Selection Heuristic**:
1. If only one candidate → select it
2. If multiple candidates:
   - **Prefer figures ABOVE the caption** (typical layout)
   - Among those, choose the one with **max horizontal overlap** with caption
   - Tie-break by **widest figure**
3. Fallback: Choose **nearest figure** by vertical distance

### Script B: Code Extractor

**Labels of Interest**: `Code` (primary), fallback to `Text` blocks near algorithm headers

**Selection Heuristic**:
1. If only one candidate → select it
2. If multiple candidates:
   - **Prefer code blocks CONTAINING the matched text** (text inside code)
   - Else prefer code blocks **BELOW the matched text** (header above code)
   - Tie-break by **largest height** (code blocks are usually tall)

## 🎨 Using Pre-computed Surya Layout

If you've already run Surya layout analysis, pass the layout JSON directly:

```python
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

# Run Surya
image = Image.open("page.png")
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)
layout_predictions = layout_predictor([image])

# Convert to dict
layout_dict = {
    'bboxes': [
        {
            'bbox': box.bbox,
            'label': box.label,
            'confidence': box.confidence,
            'position': box.position
        }
        for box in layout_predictions[0].bboxes
    ],
    'image_bbox': layout_predictions[0].image_bbox
}

# Use with extractor
result = extractor.extract_figure(
    pdf_path="paper.pdf",
    query_text="Figure 1",
    output_dir="./output",
    layout_json=layout_dict  # Pass pre-computed layout
)
```

## 📊 Layout JSON Format

The scripts work with Surya layout JSON in this format:

```json
{
  "bboxes": [
    {
      "bbox": [x1, y1, x2, y2],
      "label": "Figure",
      "confidence": 0.95,
      "position": 0
    }
  ],
  "image_bbox": [0.0, 0.0, 2550.0, 3300.0]
}
```

**Supported Labels**:
- Figure extractor: `Figure`, `Picture`
- Code extractor: `Code`, (fallback: `Text` near algorithm headers)

## 🔧 Coordinate Handling

The scripts automatically handle coordinate conversion:
- **PDF coordinates**: Origin at bottom-left, Y increases upward
- **Image coordinates**: Origin at top-left, Y increases downward

Conversion includes:
- Scaling from 72 DPI (PDF) to your chosen DPI
- Y-axis flipping
- Small padding (6-8 pixels) for robust comparisons

## 📚 Examples

See `extraction_demo.py` for comprehensive examples:

1. **Extract with pre-computed layout** - Use existing Surya output
2. **Extract with auto-run Surya** - Let script run Surya automatically
3. **Extract code blocks** - Find and extract algorithms
4. **Batch processing** - Extract multiple items at once
5. **Direct Surya integration** - Work with raw Surya objects

## 🛠️ Utilities

`surya_json_utils.py` provides helpful functions:

- `save_layout_to_json()` - Save Surya output to JSON
- `load_layout_from_json()` - Load saved layout
- `filter_boxes_by_label()` - Filter by label type
- `get_boxes_in_region()` - Find boxes overlapping a region
- `visualize_layout_boxes()` - Draw boxes on image
- `analyze_layout_statistics()` - Get layout stats
- `print_layout_summary()` - Print human-readable summary

## ⚠️ Edge Cases Handled

- ❌ **No text match** → Returns `selected_region=None` with note
- 🔀 **Multiple text matches** → Chooses highest scoring match
- ❓ **Multiple candidate regions** → Uses spatial heuristics, logs alternatives
- 🎯 **Close competition** → Tie-break by size or distance
- 📏 **Coordinate overflow** → Clamps to image bounds


## 🤝 Contributing

Each script is modular with clearly separated steps:
1. **Semantic text search** - Find query text in PDF
2. **Page rasterization** - Convert page to image
3. **Layout JSON** - Get or compute layout
4. **Candidate filtering** - Find relevant regions by label
5. **Region selection** - Apply spatial heuristics
6. **Crop and save** - Extract and save final image

Modify individual steps without affecting others!


## 🙏 Acknowledgments

- Uses [Surya](https://github.com/VikParuchuri/surya) for layout analysis
- Uses [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
