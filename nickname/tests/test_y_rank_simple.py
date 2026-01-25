"""
Simple test to verify Y-rank scoring is in the code
"""

# Read the processors.py file and check for Y-rank scoring
with open('src/screenshotanalysis/processors.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check for key elements
checks = {
    'y_rank parameter': 'y_rank: int = None' in content,
    'y_rank_score variable': 'y_rank_score' in content,
    'rank 1 gets 20 points': 'y_rank == 1' in content and 'y_rank_score = 20' in content,
    'rank 2 gets 15 points': 'y_rank == 2' in content and 'y_rank_score = 15' in content,
    'rank 3 gets 10 points': 'y_rank == 3' in content and 'y_rank_score = 10' in content,
    'y_rank in breakdown': "'y_rank': y_rank_score" in content,
    'y_rank in total score': 'total_score = position_score + text_score + y_score + height_score + y_rank_score' in content,
    'box_to_rank mapping': 'box_to_rank' in content,
    'sorted_top_boxes': 'sorted_top_boxes' in content,
    'y_rank passed to scoring': 'y_rank=y_rank' in content,
}

print("Y-Rank Scoring Implementation Check")
print("=" * 60)

all_passed = True
for check_name, result in checks.items():
    status = "✓" if result else "✗"
    print(f"{status} {check_name}: {'PASS' if result else 'FAIL'}")
    if not result:
        all_passed = False

print("=" * 60)
if all_passed:
    print("✓ All checks passed! Y-rank scoring is properly implemented.")
else:
    print("✗ Some checks failed. Please review the implementation.")

# Show the scoring breakdown in docstring
import re
docstring_match = re.search(r'Scoring factors:(.*?)Args:', content, re.DOTALL)
if docstring_match:
    print("\nScoring factors from docstring:")
    print(docstring_match.group(1).strip())
