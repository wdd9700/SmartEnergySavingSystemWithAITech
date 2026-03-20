"""
文档加载和解析模块

支持多种文档格式的加载和解析：
- Markdown (.md)
- PDF (.pdf)
- 纯文本 (.txt)
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    文档数据结构
    
    Attributes:
        content: 文档内容
        metadata: 文档元数据
        doc_id: 文档唯一标识
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理，生成文档ID"""
        if self.doc_id is None:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class DocumentChunk:
    """
    文档分块数据结构
    
    Attributes:
        content: 块内容
        doc_id: 所属文档ID
        chunk_id: 块唯一标识
        metadata: 块元数据
        start_pos: 在原文档中的起始位置
        end_pos: 在原文档中的结束位置
    """
    content: str
    doc_id: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_pos: int = 0
    end_pos: int = 0


class BaseParser(ABC):
    """文档解析器抽象基类"""
    
    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """
        解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document对象
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        检查是否支持该文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否支持
        """
        pass


class MarkdownParser(BaseParser):
    """Markdown文档解析器"""
    
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.md')
    
    def parse(self, file_path: str) -> Document:
        """
        解析Markdown文档
        
        提取内容并解析YAML frontmatter元数据
        """
        path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析YAML frontmatter
        metadata = self._extract_metadata(content)
        
        # 移除frontmatter获取纯内容
        clean_content = self._remove_frontmatter(content)
        
        # 提取标题
        title = self._extract_title(clean_content)
        
        # 提取章节
        sections = self._extract_sections(clean_content)
        
        metadata.update({
            'file_path': str(file_path),
            'file_name': path.name,
            'file_type': 'markdown',
            'title': title,
            'sections': sections,
            'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        })
        
        return Document(
            content=clean_content,
            metadata=metadata
        )
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """提取YAML frontmatter元数据"""
        metadata = {}
        pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            try:
                import yaml
                metadata = yaml.safe_load(match.group(1)) or {}
            except ImportError:
                logger.warning("PyYAML not installed, skipping YAML frontmatter parsing")
            except Exception as e:
                logger.warning(f"Failed to parse YAML frontmatter: {e}")
        
        return metadata
    
    def _remove_frontmatter(self, content: str) -> str:
        """移除YAML frontmatter"""
        pattern = r'^---\s*\n.*?\n---\s*\n'
        return re.sub(pattern, '', content, count=1, flags=re.DOTALL).strip()
    
    def _extract_title(self, content: str) -> str:
        """提取文档标题（第一个#开头的行）"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else "Untitled"
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """提取文档章节结构"""
        sections = []
        pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({
                'level': level,
                'title': title,
                'position': match.start()
            })
        
        return sections


class TextParser(BaseParser):
    """纯文本解析器"""
    
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.txt')
    
    def parse(self, file_path: str) -> Document:
        """解析纯文本文档"""
        path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试提取第一行作为标题
        lines = content.strip().split('\n')
        title = lines[0][:100] if lines else "Untitled"
        
        metadata = {
            'file_path': str(file_path),
            'file_name': path.name,
            'file_type': 'text',
            'title': title,
            'line_count': len(lines),
            'char_count': len(content),
            'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        
        return Document(
            content=content,
            metadata=metadata
        )


class PDFParser(BaseParser):
    """PDF文档解析器"""
    
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def parse(self, file_path: str) -> Document:
        """
        解析PDF文档
        
        需要安装PyPDF2或pdfplumber库
        """
        path = Path(file_path)
        
        # 尝试使用pdfplumber（效果更好）
        try:
            import pdfplumber
            return self._parse_with_pdfplumber(file_path, path)
        except ImportError:
            pass
        
        # 回退到PyPDF2
        try:
            import PyPDF2
            return self._parse_with_pypdf2(file_path, path)
        except ImportError:
            raise ImportError(
                "PDF parsing requires pdfplumber or PyPDF2. "
                "Install with: pip install pdfplumber"
            )
    
    def _parse_with_pdfplumber(self, file_path: str, path: Path) -> Document:
        """使用pdfplumber解析PDF"""
        content_parts = []
        page_count = 0
        
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    content_parts.append(f"\n--- Page {i + 1} ---\n{text}")
        
        content = '\n'.join(content_parts)
        
        metadata = {
            'file_path': str(file_path),
            'file_name': path.name,
            'file_type': 'pdf',
            'page_count': page_count,
            'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        
        return Document(content=content, metadata=metadata)
    
    def _parse_with_pypdf2(self, file_path: str, path: Path) -> Document:
        """使用PyPDF2解析PDF"""
        import PyPDF2
        
        content_parts = []
        page_count = 0
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    content_parts.append(f"\n--- Page {i + 1} ---\n{text}")
        
        content = '\n'.join(content_parts)
        
        metadata = {
            'file_path': str(file_path),
            'file_name': path.name,
            'file_type': 'pdf',
            'page_count': page_count,
            'modified_time': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        
        return Document(content=content, metadata=metadata)


class DocumentLoader:
    """
    文档加载器
    
    统一接口加载和解析多种格式的文档。
    
    Attributes:
        parsers: 解析器列表
        chunk_size: 文档分块大小
        chunk_overlap: 分块重叠大小
    
    Example:
        >>> loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
        >>> doc = loader.load("document.md")
        >>> chunks = loader.chunk_document(doc)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        初始化文档加载器
        
        Args:
            chunk_size: 文档分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 注册解析器
        self.parsers: List[BaseParser] = [
            MarkdownParser(),
            PDFParser(),
            TextParser(),
        ]
    
    def load(self, file_path: str) -> Document:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件类型
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 查找合适的解析器
        for parser in self.parsers:
            if parser.supports(file_path):
                logger.info(f"Loading document with {parser.__class__.__name__}: {file_path}")
                return parser.parse(file_path)
        
        raise ValueError(f"Unsupported file type: {file_path}")
    
    def load_directory(
        self,
        directory: str,
        pattern: Optional[str] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        加载目录中的所有文档
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式（如 "*.md"）
            recursive: 是否递归子目录
            
        Returns:
            Document对象列表
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        
        # 支持的扩展名
        supported_exts = {'.md', '.txt', '.pdf'}
        
        if recursive:
            files = directory.rglob('*')
        else:
            files = directory.iterdir()
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # 检查扩展名
            if file_path.suffix.lower() not in supported_exts:
                continue
            
            # 检查模式
            if pattern and not file_path.match(pattern):
                continue
            
            try:
                doc = self.load(str(file_path))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        将文档分块
        
        使用滑动窗口方法，保持语义完整性（尽量在段落边界分块）。
        
        Args:
            document: 文档对象
            
        Returns:
            文档块列表
        """
        content = document.content
        chunks = []
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_size = len(paragraph)
            
            # 如果当前段落加上已有内容超过块大小，保存当前块
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk_id = f"{document.doc_id}_{chunk_index}"
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    metadata=document.metadata.copy(),
                    start_pos=position,
                    end_pos=position + len(chunk_text)
                ))
                
                # 滑动窗口：保留重叠部分
                overlap_size = 0
                overlap_chunks = []
                
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= self.chunk_overlap:
                        overlap_chunks.insert(0, p)
                        overlap_size += len(p) + 2  # +2 for '\n\n'
                    else:
                        break
                
                current_chunk = overlap_chunks
                current_size = overlap_size
                chunk_index += 1
                position += len(chunk_text) - overlap_size
            
            current_chunk.append(paragraph)
            current_size += paragraph_size + 2  # +2 for '\n\n'
        
        # 处理剩余的段落
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = f"{document.doc_id}_{chunk_index}"
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                doc_id=document.doc_id,
                chunk_id=chunk_id,
                metadata=document.metadata.copy(),
                start_pos=position,
                end_pos=position + len(chunk_text)
            ))
        
        logger.info(f"Document {document.doc_id} chunked into {len(chunks)} chunks")
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        批量分块多个文档
        
        Args:
            documents: 文档列表
            
        Returns:
            所有文档块列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
