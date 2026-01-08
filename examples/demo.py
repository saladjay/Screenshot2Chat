#!/usr/bin/env python3
"""
PP-DocLayoutV2聊天内容定位示例
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chat_layout_analyzer import ChatLayoutAnalyzer, LayoutVisualizer

def demo_single_image():
    """单张聊天截图分析示例"""
    print("=== 单张聊天截图分析演示 ===")
    
    # 初始化分析器（使用均衡模型）
    analyzer = ChatLayoutAnalyzer(
        model_name="PP-DocLayout-M",
        layout_nms=True,
        threshold=0.5
    )
    
    # 分析聊天截图
    image_path = "path/to/your/chat_screenshot.png"  # 请替换为实际路径
    result = analyzer.analyze_chat_screenshot(image_path)
    
    if result['success']:
        print(f"分析成功！发现 {result['total_messages']} 条消息")
        print(f"图像尺寸: {result['image_size']}")
        
        # 打印分析摘要
        summary = result['analysis_summary']
        print(f"总元素数: {summary['total_elements']}")
        print("发现的类别:")
        for category, count in summary['categories_found'].items():
            print(f"  {category}: {count}")
            
        # 可视化结果
        visualizer = LayoutVisualizer()
        visualizer.draw_layout(
            image_path, 
            result, 
            "chat_analysis_result.png"
        )
        
    else:
        print(f"分析失败: {result['error']}")

def demo_batch_analysis():
    """批量分析示例"""
    print("\n=== 批量聊天截图分析演示 ===")
    
    analyzer = ChatLayoutAnalyzer(model_name="PP-DocLayout-M")
    
    # 假设有一个包含多张截图的目录
    image_directory = "path/to/chat/screenshots"  # 请替换为实际路径
    image_files = [
        os.path.join(image_directory, f) 
        for f in os.listdir(image_directory) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:3]  # 限制处理前3张
    
    if image_files:
        session_result = analyzer.analyze_chat_session(image_files)
        print(f"会话分析完成:")
        print(f"总图片数: {session_result['session_summary']['total_images']}")
        print(f"成功分析: {session_result['session_summary']['successful_analyses']}")
        print(f"总消息数: {session_result['session_summary']['total_messages']}")
    else:
        print("未找到聊天截图文件")

if __name__ == "__main__":
    # 演示单张图片分析
    demo_single_image()
    
    # 演示批量分析
    demo_batch_analysis()