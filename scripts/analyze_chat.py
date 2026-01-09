#!/usr/bin/env python3
"""
命令行聊天内容分析工具
"""

import argparse
from screenshotanalysis import ChatLayoutAnalyzer, LayoutVisualizer

def main():
    parser = argparse.ArgumentParser(description='聊天内容布局分析工具')
    parser.add_argument('image_path', help='聊天截图路径')
    parser.add_argument('--model', default='PP-DocLayout-M', 
                       choices=['PP-DocLayout-L', 'PP-DocLayout-M', 'PP-DocLayout-S'],
                       help='选择模型版本')
    parser.add_argument('--output', '-o', help='结果输出路径')
    parser.add_argument('--batch', action='store_true', 
                       help='批量处理目录中的所有图片')
    
    args = parser.parse_args()
    
    analyzer = ChatLayoutAnalyzer(model_name=args.model)
    visualizer = LayoutVisualizer()
    
    if args.batch:
        # 批量处理模式
        import os
        from screenshotanalysis.utils import get_image_files
        
        image_files = get_image_files(args.image_path)
        results = analyzer.analyze_chat_session(image_files)
        
        # 保存批量处理结果
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
    else:
        # 单张图片处理模式
        result = analyzer.analyze_chat_screenshot(args.image_path)
        
        if result['success']:
            print(f"分析完成！发现 {result['total_messages']} 条消息")
            
            # 可视化结果
            if args.output:
                visualizer.draw_layout(args.image_path, result, args.output)
            else:
                visualizer.draw_layout(args.image_path, result, 
                                      "chat_analysis_result.png")
        else:
            print(f"分析失败: {result['error']}")

if __name__ == "__main__":
    main()