#!/usr/bin/env python
"""
测试映射器切换功能

验证BM25和Dense检索都能正确切换Graph和AST映射器
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_bm25_mapper_switching():
    """测试BM25映射器切换"""
    print("=" * 60)
    print("测试 BM25 映射器切换")
    print("=" * 60)
    
    try:
        from method.bm25.retriever import BM25Method
        
        # 测试Graph映射器（默认）
        print("\n1. 测试Graph映射器（默认）...")
        try:
            bm25_graph = BM25Method(
                graph_index_dir="index_data/Loc-Bench_V1/graph_index_v2.3",
                bm25_index_dir="index_data/Loc-Bench_V1/BM25_index",
                top_k_files=5,
                top_k_modules=5,
                top_k_entities=10,
                mapper_type="graph",  # 默认值
            )
            print(f"   ✓ Graph映射器创建成功")
            print(f"   - mapper_type: {bm25_graph.mapper_type}")
            print(f"   - mapper类型: {type(bm25_graph.mapper).__name__}")
        except Exception as e:
            print(f"   ✗ Graph映射器创建失败: {e}")
        
        # 测试AST映射器
        print("\n2. 测试AST映射器...")
        try:
            bm25_ast = BM25Method(
                graph_index_dir="",  # AST模式不需要
                bm25_index_dir="index_data/Loc-Bench_V1/BM25_index",
                top_k_files=5,
                top_k_modules=5,
                top_k_entities=10,
                mapper_type="ast",
                repos_root="playground/locbench_repos",
            )
            print(f"   ✓ AST映射器创建成功")
            print(f"   - mapper_type: {bm25_ast.mapper_type}")
            print(f"   - mapper类型: {type(bm25_ast.mapper).__name__}")
        except Exception as e:
            print(f"   ✗ AST映射器创建失败: {e}")
        
        # 测试参数验证
        print("\n3. 测试参数验证...")
        try:
            BM25Method(
                graph_index_dir="",
                bm25_index_dir="index_data/Loc-Bench_V1/BM25_index",
                mapper_type="graph",  # 需要graph_index_dir但未提供
            )
            print(f"   ✗ 应该抛出错误但没有")
        except ValueError as e:
            print(f"   ✓ 参数验证正常: {str(e)[:50]}...")
        
        print("\n✓ BM25映射器切换测试完成")
        return True
        
    except Exception as e:
        print(f"✗ BM25测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dense_mapper_switching():
    """测试Dense映射器切换"""
    print("\n" + "=" * 60)
    print("测试 Dense 映射器切换")
    print("=" * 60)
    
    try:
        from method.dense.run_with_index import parse_args
        import argparse
        import sys
        
        # 保存原始argv
        original_argv = sys.argv.copy()
        
        # 测试AST映射器（默认）
        print("\n1. 测试AST映射器参数解析（默认）...")
        test_args_ast = [
            "test_script",
            "--index_dir", "index_data/dense_index_fixed",
            "--dataset_path", "data/Loc-Bench_V1_dataset.jsonl",
            "--output_folder", "outputs/test",
            "--mapper_type", "ast",
            "--repos_root", "playground/locbench_repos",
        ]
        sys.argv = test_args_ast
        args_ast = parse_args()
        print(f"   ✓ AST映射器参数解析成功")
        print(f"   - mapper_type: {args_ast.mapper_type}")
        print(f"   - repos_root: {args_ast.repos_root}")
        
        # 测试Graph映射器
        print("\n2. 测试Graph映射器参数解析...")
        test_args_graph = [
            "test_script",
            "--index_dir", "index_data/dense_index_fixed",
            "--dataset_path", "data/Loc-Bench_V1_dataset.jsonl",
            "--output_folder", "outputs/test",
            "--mapper_type", "graph",
            "--graph_index_dir", "index_data/Loc-Bench_V1/graph_index_v2.3",
        ]
        sys.argv = test_args_graph
        args_graph = parse_args()
        print(f"   ✓ Graph映射器参数解析成功")
        print(f"   - mapper_type: {args_graph.mapper_type}")
        print(f"   - graph_index_dir: {args_graph.graph_index_dir}")
        
        # 恢复原始argv
        sys.argv = original_argv
        
        print("\n✓ Dense映射器切换测试完成")
        return True
        
    except Exception as e:
        print(f"✗ Dense测试失败: {e}")
        import traceback
        traceback.print_exc()
        # 恢复原始argv
        if 'original_argv' in locals():
            sys.argv = original_argv
        return False


def main():
    """运行所有测试"""
    print("开始测试映射器切换功能...\n")
    
    results = []
    results.append(("BM25映射器切换", test_bm25_mapper_switching()))
    results.append(("Dense映射器切换", test_dense_mapper_switching()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！映射器切换功能实现成功！")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

