import os
from pathlib import Path
import sys
from typing import Iterator, Optional, List, Tuple
import tree_sitter
import tree_sitter_go as tsgo
from tree_sitter import Language, Parser, Tree, Node, Query
from .base import BaseChunkStrategy
from ..models.code_chunk import CodeChunk

class GolangChunkStrategy(BaseChunkStrategy):
    """基于tree-sitter的Go语言代码切片策略
    
    特点：
    1. 基于AST的智能分块
    2. 保持函数和方法的完整性
    3. 包含上下文信息（如包名、导入等）
    4. 支持注释关联
    5. 支持变量和常量定义
    """
    
    def __init__(self, max_chunk_size: int = 300):
        """初始化Go语言切片策略
        
        Args:
            max_chunk_size: 最大块大小（行数），超过此大小的函数将被拆分
        """
        # 初始化tree-sitter
        GO_LANGUAGE = Language(tsgo.language())
        self.parser = Parser(GO_LANGUAGE)
        self.language = GO_LANGUAGE
        self.max_chunk_size = max_chunk_size
        
        # 预编译查询
        self.package_query = Query(
            self.language,
            """
            (package_clause
                (package_identifier) @package.name)
            """
        )
        
        self.import_query = Query(
            self.language,
            """
            (import_declaration
                (import_spec_list
                    (import_spec) @import))
            """
        )
        
        self.type_query = Query(
            self.language,
            """
            (type_declaration
                (type_spec) @type)
            """
        )
        
        self.var_const_query = Query(
            self.language,
            """
            (var_declaration) @var
            (const_declaration) @const
            """
        )
        
        self.func_query = Query(
            self.language,
            """
            (function_declaration) @function
            (method_declaration) @method
            """
        )
        
    def _get_node_text(self, node: Node, content: str) -> str:
        """获取节点的文本内容"""
        return content[node.start_byte:node.end_byte]
        
    def _get_node_lines(self, node: Node) -> Tuple[int, int]:
        """获取节点的起始和结束行号"""
        return node.start_point[0] + 1, node.end_point[0] + 1
        
    def _find_imports(self, file_path: Path, tree: Tree):
        """查找导入声明"""
        for match in self.import_query.matches(tree.root_node):
            imports = []
            start_line = sys.maxsize
            end_line = 0
            for capture, nodes in match[1].items():
                for node in nodes:
                    imports.append(str(node.text))
                    start_line = min(start_line, node.start_point[0] + 1)
                    end_line = max(end_line, node.end_point[0] + 1)
            yield CodeChunk(
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                content="\n".join(imports)
            )
        
    def _find_associated_comments(self, node: Node, content: str) -> List[str]:
        """查找与节点关联的注释"""
        comments = []
        current = node.prev_sibling
        while current and current.type == "comment":
            comments.insert(0, self._get_node_text(current, content))
            current = current.prev_sibling
        return comments
        
    def _find_type_specs(self, file_path: Path, tree: Tree):
        """查找类型定义"""
        for match in self.type_query.matches(tree.root_node):
            for capture, nodes in match[1].items():
                for node in nodes:
                    yield CodeChunk(
                        file_path=str(file_path),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content=str(node.text)
                    )
        
    def _find_var_const_block(
        self,
        file_path: Path,
        tree: Tree,
    ) -> Iterator[CodeChunk]:
        """处理变量或常量声明块"""
        for match in self.var_const_query.matches(tree.root_node):
            for capture, nodes in match[1].items():
                for node in nodes:
                    yield CodeChunk(
                        file_path=str(file_path),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content=str(node.text)
                    )
    
    def _find_function_declaration(
        self,
        file_path: Path,
        tree: Tree,
    ) -> Iterator[CodeChunk]:
        """查找函数声明"""
        for match in self.func_query.matches(tree.root_node):
            for capture, nodes in match[1].items():
                for node in nodes:
                    yield CodeChunk(
                        file_path=str(file_path),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content=str(node.text)
                    )
        

    def _split_large_function(
        self,
        file_path: Path,
        node: Node,
        content: str,
        package_info: Optional[str],
        imports: List[str]
    ) -> Iterator[CodeChunk]:
        """拆分大型函数"""
        start_line, end_line = self._get_node_lines(node)
        total_lines = end_line - start_line + 1
        
        if total_lines <= self.max_chunk_size:
            yield self._create_chunk(file_path, node, content, package_info, imports)
            return
            
        # 查找函数体中的语句
        body_node = None
        for child in node.children:
            if child.type == "block":
                body_node = child
                break
                
        if not body_node:
            yield self._create_chunk(file_path, node, content, package_info, imports)
            return
            
        # 获取函数签名
        signature = self._get_node_text(node, content)[:body_node.start_byte - node.start_byte].decode('utf-8')
        
        # 按语句分组
        current_chunk = []
        current_lines = 0
        
        for stmt in body_node.children:
            if stmt.type == "{" or stmt.type == "}":
                continue
                
            stmt_start, stmt_end = self._get_node_lines(stmt)
            stmt_lines = stmt_end - stmt_start + 1
            
            if current_lines + stmt_lines > self.max_chunk_size and current_chunk:
                # 创建当前块
                chunk_content = signature + " {\n" + "\n".join(current_chunk) + "\n}"
                yield CodeChunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content
                )
                current_chunk = []
                current_lines = 0
                
            current_chunk.append(self._get_node_text(stmt, content))
            current_lines += stmt_lines
            
        # 处理最后一个块
        if current_chunk:
            chunk_content = signature + " {\n" + "\n".join(current_chunk) + "\n}"
            yield CodeChunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                content=chunk_content
            )
            
    def chunk_file(self, file_path: Path, content: str) -> Iterator[CodeChunk]:
        """对Go文件进行分块
        
        策略：
        1. 解析文件获取AST
        2. 提取包信息和导入
        3. 提取类型定义
        4. 处理变量和常量定义
        5. 遍历所有函数和方法定义
        6. 对大型函数进行智能拆分
        """
        # 解析文件
        tree = self.parser.parse(content.encode('utf-8'))
        
        # 获取包信息和导入
        yield from self._find_imports(file_path, tree)
        yield from self._find_type_specs(file_path, tree)
        yield from self._find_var_const_block(file_path, tree)
        yield from self._find_function_declaration(file_path, tree)
