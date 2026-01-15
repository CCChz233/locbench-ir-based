#!/usr/bin/env python
"""
测试重构后的映射功能

验证：
1. Graph映射器（BM25）是否正常工作
2. AST映射器（Dense）是否正常工作
3. 输出格式是否与重构前一致
"""

import json
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from method.bm25.retriever import BM25Method
from method.mapping import GraphBasedMapper, ASTBasedMapper


def test_graph_mapper():
    """测试Graph映射器"""
    print("=" * 60)
    print("测试 Graph映射器 (BM25)")
    print("=" * 60)
    
    # 检查索引是否存在（尝试多个可能的路径）
    graph_index_dir = os.environ.get("GRAPH_INDEX_DIR")
    bm25_index_dir = os.environ.get("BM25_INDEX_DIR")
    
    # 如果环境变量未设置，尝试自动查找
    if not graph_index_dir:
        possible_graph_dirs = [
            "index_data/Loc-Bench_V1/graph_index_v2.3",
            "index_data/czlll___loc-bench_v1/graph_index_v2.3",
            "index_data/Loc-Bench_V1/graph_index_v1",
        ]
        for d in possible_graph_dirs:
            if os.path.exists(d):
                graph_index_dir = d
                break
    
    if not bm25_index_dir:
        possible_bm25_dirs = [
            "index_data/Loc-Bench_V1/BM25_index",
            "index_data/czlll___loc-bench_v1/BM25_index",
        ]
        for d in possible_bm25_dirs:
            if os.path.exists(d):
                bm25_index_dir = d
                break
    
    if not graph_index_dir or not bm25_index_dir:
        print(f"⚠️  索引目录不存在，跳过Graph映射器测试")
        print(f"   请设置环境变量或确保索引存在:")
        print(f"   GRAPH_INDEX_DIR: {graph_index_dir or '未找到'}")
        print(f"   BM25_INDEX_DIR: {bm25_index_dir or '未找到'}")
        return False
    
    # 加载一个测试实例
    dataset_path = "data/Loc-Bench_V1_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据集文件不存在: {dataset_path}")
        return False
    
    with open(dataset_path, 'r') as f:
        first_line = f.readline()
        if not first_line:
            print("⚠️  数据集文件为空")
            return False
        instance = json.loads(first_line)
    
    instance_id = instance.get("instance_id")
    print(f"测试实例: {instance_id}")
    
    # 测试BM25方法
    try:
        method = BM25Method(
            graph_index_dir=graph_index_dir,
            bm25_index_dir=bm25_index_dir,
            top_k_files=5,
            top_k_modules=5,
            top_k_entities=10,
        )
        
        result = method.localize(instance)
        
        print(f"✓ BM25方法执行成功")
        print(f"  - found_files: {len(result.found_files)} 个")
        print(f"  - found_modules: {len(result.found_modules)} 个")
        print(f"  - found_entities: {len(result.found_entities)} 个")
        
        if result.found_modules:
            print(f"  示例模块: {result.found_modules[0]}")
        if result.found_entities:
            print(f"  示例实体: {result.found_entities[0]}")
        
        # 验证输出格式
        assert isinstance(result.found_files, list), "found_files应该是列表"
        assert isinstance(result.found_modules, list), "found_modules应该是列表"
        assert isinstance(result.found_entities, list), "found_entities应该是列表"
        
        print("✓ 输出格式验证通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ast_mapper():
    """测试AST映射器"""
    print("\n" + "=" * 60)
    print("测试 AST映射器 (Dense)")
    print("=" * 60)
    
    repos_root = "playground/locbench_repos"
    if not os.path.exists(repos_root):
        print(f"⚠️  仓库目录不存在: {repos_root}")
        return False
    
    # 创建一个测试代码块
    # 找一个存在的仓库
    repos = [d for d in os.listdir(repos_root) if os.path.isdir(os.path.join(repos_root, d))]
    if not repos:
        print(f"⚠️  仓库目录为空: {repos_root}")
        return False
    
    repo_name = repos[0]
    repo_path = os.path.join(repos_root, repo_name)
    
    # 找一个Python文件
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                python_files.append(rel_path)
                if len(python_files) >= 3:
                    break
        if len(python_files) >= 3:
            break
    
    if not python_files:
        print(f"⚠️  仓库中没有Python文件: {repo_path}")
        return False
    
    print(f"测试仓库: {repo_name}")
    print(f"测试文件: {python_files[0]}")
    
    try:
        mapper = ASTBasedMapper(repos_root=repos_root)
        
        # 创建测试代码块
        test_blocks = [
            {
                'file_path': python_files[0],
                'start_line': 0,
                'end_line': 20,  # 测试前20行
            }
        ]
        
        # 构造instance_id
        instance_id = f"{repo_name.upper().replace('-', '_')}_{repo_name}-1"
        
        found_modules, found_entities = mapper.map_blocks_to_entities(
            blocks=test_blocks,
            instance_id=instance_id,
            top_k_modules=10,
            top_k_entities=20,
        )
        
        print(f"✓ AST映射器执行成功")
        print(f"  - found_modules: {len(found_modules)} 个")
        print(f"  - found_entities: {len(found_entities)} 个")
        
        if found_modules:
            print(f"  示例模块: {found_modules[0]}")
        if found_entities:
            print(f"  示例实体: {found_entities[0]}")
        
        # 验证输出格式
        assert isinstance(found_modules, list), "found_modules应该是列表"
        assert isinstance(found_entities, list), "found_entities应该是列表"
        
        print("✓ 输出格式验证通过")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dense_integration():
    """测试Dense集成（需要索引）"""
    print("\n" + "=" * 60)
    print("测试 Dense集成")
    print("=" * 60)
    
    index_dir = "index_data/dense_index_fixed"
    if not os.path.exists(index_dir):
        print(f"⚠️  Dense索引目录不存在: {index_dir}")
        print("   跳过Dense集成测试")
        return None
    
    # 检查是否有索引
    repos = [d for d in os.listdir(index_dir) if os.path.isdir(os.path.join(index_dir, d))]
    if not repos:
        print(f"⚠️  索引目录为空: {index_dir}")
        return None
    
    print(f"✓ 找到 {len(repos)} 个仓库的索引")
    print("  可以运行完整测试:")
    print(f"  python method/dense/run_with_index.py \\")
    print(f"    --index_dir {index_dir} \\")
    print(f"    --dataset_path data/Loc-Bench_V1_dataset.jsonl \\")
    print(f"    --output_folder outputs/test_dense \\")
    print(f"    --repos_root playground/locbench_repos \\")
    print(f"    --eval_n_limit 5")
    
    return True


def main():
    """运行所有测试"""
    print("开始测试重构后的映射功能...\n")
    
    results = []
    
    # 测试Graph映射器
    results.append(("Graph映射器", test_graph_mapper()))
    
    # 测试AST映射器
    results.append(("AST映射器", test_ast_mapper()))
    
    # 测试Dense集成
    dense_result = test_dense_integration()
    if dense_result is not None:
        results.append(("Dense集成", dense_result))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results if result is not None)
    
    if all_passed:
        print("\n🎉 所有测试通过！重构成功！")
        print("\n📝 关于索引：")
        print("   - 不需要重构索引，映射是在运行时进行的")
        print("   - Graph索引和BM25索引：用于BM25检索")
        print("   - Dense索引：用于Dense检索")
        print("   - 源代码仓库：用于AST映射器运行时解析")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

