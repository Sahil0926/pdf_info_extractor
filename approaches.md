# PDF Element Extraction: Approaches and Learnings

This document summarizes the different approaches I explored for extracting elements (images, algorithms, and associated text) from PDF files. 
It covers the initial attempts, their limitations, the final approach I selected, and the improvements I would like to incorporate in the future.

---

## 1. Text-Based Approaches

- **Method**:  
  - Used semantic search to locate the relevant region in the PDF.  
  - Attempted to merge detected lines and nearby text blocks.  

- **Challenges**:  
  - Required applying a large number of heuristics.  
  - Handling edge cases and text merging became too complex and error-prone.  
  - Overall, not scalable or feasible in practice.  

---

## 2. PDF Dictionary Tree and XObject Exploration

- **Method**:  
  - Explored the internal PDF structure (dictionary tree).  
  - Investigated `XObject` paths to extract images directly.  
  - Tried to combine multiple image objects into a single output image.  

- **Challenges**:  
  - Images were often split into groups of smaller objects, making merging complex.  
  - The process became lengthy and difficult to manage.  
  - For algorithms, text extraction still required merging and ordering, which remained tricky.  

---

## 3. Hybrid Approach: Similarity Search + Layout Model

- **Method**:  
  - Performed similarity search on user queries to identify the most relevant page number and region.  
  - Converted the selected PDF page into an image.  
  - Applied a layout detection model (experimented with different options, finally chose **Surya**) to identify and tag regions (figures, tables, paragraphs, etc.).  
  - Extracted the final elements based on these tagged boxes.  

- **Reason for Choosing Surya**:  
  - Promising outputs observed in their repository.  
  - Clear tagging of elements, which reduced manual heuristics.  

---

## 4. Final Selected Approach

The chosen solution was the **hybrid pipeline**:  

1. **Similarity search** → Identify relevant page(s).  
2. **Convert page to image** → Prepare for layout analysis.  
3. **Layout model (Surya)** → Detect and tag regions.  
4. **Output extraction** → Return the relevant element(s).  

This pipeline provided the best balance between accuracy, scalability, and implementation feasibility under time and resource constraints.

---

## 5. Future Improvements

- **Better Similarity Search**  
  - Current similarity search is basic.  
  - Needs robust embeddings to handle **complex queries** from users.  

- **Confidence-Based Filtering**  
  - Layout model outputs had low confidence scores.  
  - Plan: fine-tune/train a custom model on domain-specific data.  
  - Use **confidence thresholding** to eliminate false positives.  

- **Performance Optimization**  
  - Current solution runs on CPU, leading to slow inference.  
  - Deploying on **GPU** would drastically improve speed and scalability.  

---

## 6. Conclusion

The journey involved multiple iterations:  
- Starting from text-only approaches,  
- Moving to low-level PDF object exploration,  
- Finally arriving at a hybrid similarity + layout model solution.  

While the final pipeline works effectively, future enhancements in embeddings, confidence filtering, and GPU deployment will make it more robust and production-ready.
