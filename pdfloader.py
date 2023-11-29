from abc import ABC
import json
import os
import yaml
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, TypedDict
import pdfplumber
import pandas as pd
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.document_loaders.base import BaseBlobParser, Document
from langchain.document_loaders import Blob
from langchain.schema import BaseDocumentTransformer
from tqdm import tqdm
from langchain.document_loaders import DirectoryLoader
import constants

from tableToStruct import ConvertTable

def keep_bold_chars(obj):
    if obj['object_type'] == 'char':
        return 'Bold' in obj['fontname'] and obj["size"] >= 11.999999999999943
    return True

def keep_normal_chars(obj):
    if obj['object_type'] == 'char':
        return not ('Bold' in obj['fontname'] and obj["size"] >= 11.999999999999943)
    return True

class PdfBaseElement(ABC):
    def __init__(self, page):
        self.page = page
    def __repr__(self):
        return str(self)

class PdfSectionElement(PdfBaseElement):
    def __init__(self, text, page):
        PdfBaseElement.__init__(self,page)
        self.text = text

    def __str__(self):
        return f'Section: {self.text}'


class PdfTextElement(PdfBaseElement):
    def __init__(self, text,page):
        PdfBaseElement.__init__(self,page)
        self.text = text

    def __str__(self):
        return f'Text: {self.text}'

class PdfTableElement(PdfBaseElement):
    def __init__(self, table, page):
        PdfBaseElement.__init__(self,page)
        self.table = table    
        (self.jsonData, self.jsonHeader) = ConvertTable(table)

    def __str__(self):
        return f'Table:'

error_margin = 1.0

class TextElement(TypedDict):
    text: str
    bbox: object

#convert a list of words, into a list of lines
def parseLine(words) -> list[TextElement]:
    text_lines = []
    current_line = {'text': '','bbox' : None}
    #find each line of text given all words:
    for word in words:
        if current_line['text'] and abs(current_line['bbox'][3] - word['bottom']) > error_margin:
            # If the current word is on a new line, add the current line to the result
            text_lines.append(current_line)
            current_line = {'text': '','bbox' : None}

        current_line['text'] += word['text'] + ' '

        if current_line['bbox']:
            current_line['bbox'] = [min(current_line['bbox'][0], word['x0']),
                                    min(current_line['bbox'][1], word['top']),
                                    max(current_line['bbox'][2], word['x1']),
                                    max(current_line['bbox'][3], word['bottom'])]
        else:
            current_line['bbox'] = [word["x0"], word["top"], word["x1"], word["bottom"]]

    if current_line['text']:
        text_lines.append(current_line)
    
    return text_lines


def parsePdf(path) -> List[PdfBaseElement]:
    PdfData = []

    with pdfplumber.open(path) as pdf:
        for page in tqdm(pdf.pages):
            #if page.page_number != 9:
            #    continue


            tables = page.find_tables(table_settings={})

            def withoutTable(obj):
                for table in tables:
                    if obj["x0"] > table.bbox[0] and  obj["top"] > table.bbox[1] and  obj["x1"] < table.bbox[2] and  obj["bottom"] < table.bbox[3]:
                        return False

                return True
            page = page.dedupe_chars()
            page_withoutTable = page.filter(withoutTable)
            page_sections = page_withoutTable.filter(keep_bold_chars)
            page_normal_text = page_withoutTable.filter(keep_normal_chars)
            
            section_words = page_sections.extract_words()
            TextElements = parseLine(section_words)

            for table in tables:
                res = table.extract()
                #df = pd.DataFrame(res[1:], columns=res[0])
                TextElements.append({'table': 'True', 'bbox' : table.bbox, 'data' : res})

            TextElements.sort(key=lambda x : x['bbox'][1])


            # Add TextElement and Text in between to PdfData
            for i in range(-1, len(TextElements)):
                if i >= 0:
                    if len(PdfData) > 0 and isinstance(PdfData[-1], PdfSectionElement) and not "table" in TextElements[i]:
                        PdfData[-1].text += TextElements[i]['text']
                    elif len(PdfData) > 0 and isinstance(PdfData[-1], PdfTableElement) and "table" in TextElements[i]:
                        t = PdfTableElement(TextElements[i]['data'], page.page_number)
                        if PdfData[-1].jsonHeader == t.jsonHeader:
                            PdfData[-1].jsonData.extend(t.jsonData)
                        else:
                            PdfData.append(t)
                    else:
                        if "table" in TextElements[i]:
                            PdfData.append(PdfTableElement(TextElements[i]['data'], page.page_number))
                        else:
                            PdfData.append(PdfSectionElement(TextElements[i]['text'], page.page_number))
                           
                
                def textBetween(obj):
                    return (obj["top"] >= TextElements[i]['bbox'][1] if i >= 0 else True) and (obj["bottom"] <= TextElements[i+1]['bbox'][3] if i+1 < len(TextElements) else True)
                page_textbetween = page_normal_text.filter(textBetween)
                
                inbetween_words = page_textbetween.extract_words()
                inbetween_lines = parseLine(inbetween_words)

                def line_is_not_header(line : TextElement):
                    text = line['text']
                    height = page.bbox[3] 
                    if f"{str(page.page_number)} / {str(len(pdf.pages))}" in text or (f"Seite {str(page.page_number)}" in text):
                        diff_top =  line['bbox'][1]
                        if diff_top / height < 0.1 or diff_top / height > 0.9:
                            return False
                    return True


                inbetween_text = "\n".join(map(lambda c : c['text'], filter(line_is_not_header, inbetween_lines)))
                #inbetween_text = page_textbetween.extract_text()

                if(inbetween_text.strip()):
                    PdfData.append(PdfTextElement(inbetween_text, page.page_number))
    return PdfData


class PDFElementCombiner(BaseDocumentTransformer, ABC):
    """Combine Section and PDF Text"""

    def __init__(
        self,
    ) -> None:
        """Create a new PDFElementCombiner.
        """
      

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        result = []
        lastSection = None
        for doc in documents:
            
            if doc.metadata["type"] == "PdfSectionElement":
                doc.metadata["section"] = doc.page_content.strip()
                lastSection = doc

            if lastSection != None:
                doc.metadata["section"]= lastSection.metadata["section"]

            if len(result) > 0:
                last : Document = result[-1]
                if doc.metadata["type"] == "PdfTextElement" and last.metadata["type"] == "PdfSectionElement":
                    last.page_content = (last.page_content + "\n" + doc.page_content).strip()
                else: 
                    result.append(doc)
                    
            else:
                result.append(doc)
        return result


class PDFCustomLoader(BasePDFLoader):
    """Load `PDF` files using custom `pdfplumber` analyser."""

    def __init__(
        self,
        file_path: str,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
        combiner :  BaseDocumentTransformer = PDFElementCombiner()
    ) -> None:
        """Initialize with a file path."""
        try:
            import pdfplumber  # noqa:F401
        except ImportError:
            raise ImportError(
                "pdfplumber package not found, please install it with "
                "`pip install pdfplumber`"
            )

        super().__init__(file_path, headers=headers)
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images
        self.combiner = combiner

    def load(self) -> List[Document]:
        """Load file."""

        parser = PDFCustomParser(
            text_kwargs=self.text_kwargs,
            dedupe=self.dedupe,
            extract_images=self.extract_images,
        )
        blob = Blob.from_path(self.file_path)
        return self.combiner.transform_documents(parser.parse(blob))

class PDFCustomParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
        self,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        extract_images: bool = False,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images



    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_path:
            data = parsePdf(file_path)
            
            for ele in data:

                def getDoc(text, blob, type, page):
                    return Document(
                        page_content= text,
                        metadata=dict(
                            {
                                "source": blob.source,
                                "type" : type,
                                "page" : page,

                            }
                        ),
                    )
                
                text = ""
                if isinstance(ele, PdfTableElement):
                    table_data : List[List[str | None]] = ele.table
                    if len(table_data[0]) > 5:
                        #2. to yaml
                        for row in ele.jsonData:
                            rowyaml = yaml.dump(row,allow_unicode=True)
                            yield getDoc(rowyaml, blob, "PdfTableElementRow", ele.page)
                    else:
                        data = []
                        yaml_table = ""
                        for row in ele.jsonData:
                            data.append(row)
                            yaml_table = yaml.dump(data,allow_unicode=True)
                            if len(yaml_table) > constants.MAX_DOCUMENT_CHUNK_SIZE:
                                yield getDoc(yaml_table, blob, "PdfTable", ele.page)
                                yaml_table = ""
                                data = []
                        if yaml_table:
                            yield getDoc(yaml_table, blob, "PdfTable", ele.page)
                    #    #1. convert to df
                    #    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    #    text = df.to_markdown()
                    #    yield getDoc(text, blob, ele.__class__.__name__, ele.page)
                    #    #text = df.to_json()
                    #    #text = df.to_markdown()
                else:
                    yield getDoc(ele.text, blob, ele.__class__.__name__, ele.page)
                

if __name__ == "__main__":
    path = "data/pdfs/PVO_2023_V5.pdf"
    #path = "data/pdfs/ZLO_2021_V2.pdf"
    
    #path = "data/pdfs/Curriculum-B_Inf.pdf"
    #path = "data/pdfs/CMaster_Informatik.pdf"
    # data = parsePdf(path)
    # for entry in data:
    #     print(entry)

    loader = PDFCustomLoader(path)
    docs = loader.load()

    val_path =  f"validation/dokuments{os.path.basename(path)}.html"
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    with open(val_path, "w", encoding="utf-8") as file:
        file.write("<span style=\"white-space: pre\">")#show new line in html!

        for entry in docs:
            file.write(f"\n<font color='red'>{str(entry.metadata)}</font>\n")
            file.write(f"{str(entry.page_content)}")
            

        file.write("</span>")#show new line in html!

    
    #loader = DirectoryLoader(constants.PATH_PDF, glob="**/*.pdf", loader_cls=PDFCustomLoader, show_progress=True)
    #docs = loader.load_and_split(constants.SPLITTER)
    #docsSorted = sorted(docs, key=lambda c : len(c.page_content))
    #for entry in docsSorted:
    #    print(len(entry.page_content))

