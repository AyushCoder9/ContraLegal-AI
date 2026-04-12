import fitz  # PyMuPDF

def highlight_contract_risks(input_path: str, output_path: str, flagged_clauses: list):

    doc = fitz.open(input_path) #opening the file 
    
    for item in flagged_clauses:
        clause_text = item['text']
        risk_level = item['risk']
        
        color = (1, 0, 0) if "High" in risk_level else (1, 0.8, 0)  # Define colors (RGB format 0.0 to 1.0) Red for High Risk, Yellow for Medium
        
        for page in doc:
            page_text = page.get_text()

            if clause_text[:50].lower() in page_text.lower():   # quick filter
                text_instances = page.search_for(clause_text[:100])

                seen = set()
                for inst in text_instances:
                    key = (inst.x0, inst.y0, inst.x1, inst.y1)

                    if key in seen:
                        continue

                    seen.add(key)

                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=color)
                    annot.update()
                                
    # here save the new PDF
    doc.save(output_path, garbage=4, deflate=True) 
    doc.close()
    return output_path