from abc import ABC
import json
import yaml
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence
import pdfplumber
import pandas as pd
from langchain.document_loaders.pdf import BasePDFLoader
from langchain.document_loaders.base import BaseBlobParser, Document
from langchain.document_loaders import Blob
from langchain.schema import BaseDocumentTransformer
from tqdm import tqdm

from tableToStruct import ConvertTable

def keep_bold_chars(obj):
    if obj['object_type'] == 'char':
        return 'Bold' in obj['fontname'] and obj["size"] >= 12
    return True

def keep_normal_chars(obj):
    if obj['object_type'] == 'char':
        return not ('Bold' in obj['fontname'] and obj["size"] >= 12)
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
    def __str__(self):
        return f'Table:'

def parsePdf(path) -> List[PdfBaseElement]:
    PdfData = []

    with pdfplumber.open(path) as pdf:
        for page in tqdm(pdf.pages):
           
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
            current_line = {'text': '','bbox' : None}
            error_margin = 1.0
            TextElements = []
            #find each line of text given all words:
            for word in section_words:
                if current_line['text'] and abs(current_line['bbox'][3] - word['bottom']) > error_margin:
                    # If the current word is on a new line, add the current line to the result
                    TextElements.append(current_line)
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
                TextElements.append(current_line)
            
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
                    else:
                        if "table" in TextElements[i]:
                            PdfData.append(PdfTableElement(TextElements[i]['data'], page.page_number))
                        else:
                            PdfData.append(PdfSectionElement(TextElements[i]['text'], page.page_number))
                           
                
                def textBetween(obj):
                    return (obj["top"] >= TextElements[i]['bbox'][1] if i >= 0 else True) and (obj["bottom"] <= TextElements[i+1]['bbox'][3] if i+1 < len(TextElements) else True)
                page_textbetween = page_normal_text.filter(textBetween)
                
                text = page_textbetween.extract_text()
                if(text):
                    PdfData.append(PdfTextElement(text, page.page_number))
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
                if doc.metadata["type"] != "PdfSectionElement" and last.metadata["type"] == "PdfSectionElement":
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
                                "page" : page
                            }
                        ),
                    )
                
                text = ""
                if isinstance(ele, PdfTableElement):
                    table_data : List[List[str | None]] = ele.table
                    if len(table_data[0]) > 5:
                        #2. to json
                        jsonData = ConvertTable(table_data)
                        for row in jsonData:
                            rowyaml = yaml.dump(row,allow_unicode=True)
                            rowjson = json.dumps(row,indent=4,separators=(',', ': '))
                            yield getDoc(rowyaml, blob, "PdfTableElementRow", ele.page)
                    else:
                        #1. convert to df
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        text = df.to_html()
                        yield getDoc(text, blob, ele.__class__.__name__, ele.page)
                        #text = df.to_json()
                        #text = df.to_markdown()
                else:
                    yield getDoc(ele.text, blob, ele.__class__.__name__, ele.page)
                

if __name__ == "__main__":
    path = "data/pdfs/PVO_2023_V5.pdf"
    #path = "data/pdfs/Curriculum-B_Inf.pdf"
    #path = "data/pdfs/CMaster_Informatik.pdf"
    # data = parsePdf(path)
    # for entry in data:
    #     print(entry)

    loader = PDFCustomLoader(path)
    docs = loader.load()
    for entry in docs:
        print(entry)



