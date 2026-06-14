import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx_to_file(docx_path, out_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            with open(out_path, 'w', encoding='utf-8') as f:
                for paragraph in tree.findall('.//w:p', namespace):
                    texts = [node.text for node in paragraph.findall('.//w:t', namespace) if node.text]
                    if texts:
                        f.write(''.join(texts) + '\n')
        print(f"Saved {docx_path} to {out_path}")
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")

base_dir = r'd:\Hackathons\India Runs\dataset\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge'

read_docx_to_file(r'd:\Hackathons\India Runs\IndiaRuns_guidelines.docx', 'IndiaRuns_guidelines.txt')
read_docx_to_file(f"{base_dir}\\job_description.docx", 'job_description.txt')
read_docx_to_file(f"{base_dir}\\redrob_signals_doc.docx", 'redrob_signals_doc.txt')
read_docx_to_file(f"{base_dir}\\submission_spec.docx", 'submission_spec.txt')
read_docx_to_file(f"{base_dir}\\README.docx", 'README.txt')
