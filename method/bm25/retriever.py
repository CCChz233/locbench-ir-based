"""
BM25 检索器核心逻辑

基于 BM25 算法的代码定位方法。
"""

import logging
from typing import List, Optional

from method.base import LocResult, BaseMethod
from method.mapping import GraphBasedMapper, ASTBasedMapper
from method.utils import (
    get_problem_text,
    instance_id_to_repo_name,
    load_graph_index,
    load_bm25_retriever,
    dedupe_append,
    clean_file_path,
)

logger = logging.getLogger(__name__)


class BM25Method(BaseMethod):
    """
    BM25 基线方法
    
    使用 BM25 算法检索与问题描述最相关的代码片段，
    然后从检索结果中提取文件、模块和实体信息。
    """
    
    def __init__(
        self,
        graph_index_dir: str,
        bm25_index_dir: str,
        top_k_files: int = 10,
        top_k_modules: int = 10,
        top_k_entities: int = 10,
        mapper_type: str = "graph",
        repos_root: Optional[str] = None,
    ):
        """
        初始化 BM25 方法
        
        Args:
            graph_index_dir: 图索引目录（mapper_type='graph'时必需）
            bm25_index_dir: BM25 索引目录
            top_k_files: 返回的文件数量
            top_k_modules: 返回的模块数量
            top_k_entities: 返回的实体数量
            mapper_type: 映射器类型，'graph' 或 'ast'，默认 'graph'
            repos_root: 源代码仓库根目录（mapper_type='ast'时必需）
        """
        self.graph_index_dir = graph_index_dir
        self.bm25_index_dir = bm25_index_dir
        self.top_k_files = top_k_files
        self.top_k_modules = top_k_modules
        self.top_k_entities = top_k_entities
        self.mapper_type = mapper_type
        self.repos_root = repos_root
        
        # 根据mapper_type选择映射器
        if mapper_type == "graph":
            if not self.graph_index_dir:
                raise ValueError(
                    "mapper_type='graph' 时必须提供 graph_index_dir 参数。\n"
                    "示例: BM25Method(..., graph_index_dir='index_data/.../graph_index_v2.3', mapper_type='graph')"
                )
            self.mapper = GraphBasedMapper(graph_index_dir=self.graph_index_dir)
        elif mapper_type == "ast":
            if not self.repos_root:
                raise ValueError(
                    "mapper_type='ast' 时必须提供 repos_root 参数。\n"
                    "示例: BM25Method(..., repos_root='playground/locbench_repos', mapper_type='ast')"
                )
            self.mapper = ASTBasedMapper(repos_root=self.repos_root)
        else:
            raise ValueError(f"未知的 mapper_type: {mapper_type}，必须是 'graph' 或 'ast'")
    
    @property
    def name(self) -> str:
        return "BM25"
    
    def localize(self, instance: dict) -> LocResult:
        """
        使用 BM25 进行代码定位
        
        Args:
            instance: 数据集实例
        
        Returns:
            LocResult: 定位结果
        """
        instance_id = instance["instance_id"]
        repo_name = instance_id_to_repo_name(instance_id)
        
        # 加载 BM25 检索器
        retriever = load_bm25_retriever(instance_id, self.bm25_index_dir)
        if retriever is None:
            logger.warning(f"Missing BM25 index for {instance_id}. Returning empty result.")
            return LocResult.empty(instance_id)
        
        # 设置检索数量
        max_k = max(self.top_k_files, self.top_k_modules, self.top_k_entities)
        if hasattr(retriever, "similarity_top_k"):
            retriever.similarity_top_k = max_k
        
        # 获取查询文本
        query = get_problem_text(instance)
        if not query:
            logger.warning(f"No problem statement for {instance_id}. Returning empty result.")
            return LocResult.empty(instance_id)
        
        # 执行检索
        found_files: List[str] = []
        found_modules: List[str] = []
        found_entities: List[str] = []
        
        try:
            retrieved_nodes = retriever.retrieve(query)
        except ValueError as e:
            if "corpus size should be larger than top-k" in str(e):
                logger.warning(f"Corpus too small for {instance_id}: {e}")
                return LocResult.empty(instance_id)
            raise
        
        # 处理检索结果：提取文件列表
        block_metadata_list = []
        for node in retrieved_nodes:
            file_path = node.metadata.get("file_path")
            if file_path:
                file_path = clean_file_path(file_path, repo_name)
                dedupe_append(found_files, file_path, self.top_k_files)
                # 收集代码块metadata用于映射
                block_metadata_list.append(node.metadata)
            
            # 检查是否已达到文件数量限制
            if len(found_files) >= self.top_k_files:
                break
        
        # 使用映射器提取模块和实体
        if block_metadata_list:
            found_modules, found_entities = self.mapper.map_blocks_to_entities(
                blocks=block_metadata_list,
                instance_id=instance_id,
                top_k_modules=self.top_k_modules,
                top_k_entities=self.top_k_entities,
            )
        
        return LocResult(
            instance_id=instance_id,
            found_files=found_files,
            found_modules=found_modules,
            found_entities=found_entities,
        )


def run_bm25_localization(
    instance: dict,
    graph_index_dir: str,
    bm25_index_dir: str,
    top_k_files: int = 10,
    top_k_modules: int = 10,
    top_k_entities: int = 10,
    mapper_type: str = "graph",
    repos_root: Optional[str] = None,
) -> LocResult:
    """
    运行 BM25 定位（函数式接口）
    
    Args:
        instance: 数据集实例
        graph_index_dir: 图索引目录（mapper_type='graph'时必需）
        bm25_index_dir: BM25 索引目录
        top_k_files: 返回的文件数量
        top_k_modules: 返回的模块数量
        top_k_entities: 返回的实体数量
        mapper_type: 映射器类型，'graph' 或 'ast'，默认 'graph'
        repos_root: 源代码仓库根目录（mapper_type='ast'时必需）
    
    Returns:
        LocResult: 定位结果
    """
    method = BM25Method(
        graph_index_dir=graph_index_dir,
        bm25_index_dir=bm25_index_dir,
        top_k_files=top_k_files,
        top_k_modules=top_k_modules,
        top_k_entities=top_k_entities,
        mapper_type=mapper_type,
        repos_root=repos_root,
    )
    return method.localize(instance)

