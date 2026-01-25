# Y-Direction Ranking Score Implementation

## Summary

Successfully added Y-direction ranking score to the nickname detection scoring system in `src/screenshotanalysis/processors.py`.

## Changes Made

### 1. Updated `_calculate_nickname_score` Method

Added a new scoring factor: **Y Rank Score (0-20 points)**

**Scoring Rules:**
- 1st place (topmost): 20 points
- 2nd place: 15 points
- 3rd place: 10 points
- 4th place and beyond: 0 points

**Total Scoring System (now 100 points max):**
1. Position score (0-15): Horizontal position relative to screen center
2. Text score (0-30): Whether text is system UI text
3. Y position score (0-15): Whether in top region but not extreme top
4. Height score (0-20): Font size (larger = higher score)
5. **Y rank score (0-20): Ranking based on Y position** ← NEW

### 2. Updated `extract_nicknames_smart` Method

Modified the candidate scoring loop to:
1. Sort all boxes by Y position before scoring
2. Create a mapping from box ID to Y rank
3. Pass the Y rank to the scoring function for each candidate

## Code Changes

### File: `src/screenshotanalysis/processors.py`

#### Change 1: Added `y_rank` parameter to scoring function (line ~1662)

```python
def _calculate_nickname_score(self, box: TextBox, text: str, 
                              screen_width: int, screen_height: int,
                              y_rank: int = None):  # ← NEW PARAMETER
```

#### Change 2: Added Y rank scoring logic (line ~1730)

```python
# 5. Y rank score (0-20): ranking based on Y position
if y_rank is not None:
    if y_rank == 1:
        y_rank_score = 20
    elif y_rank == 2:
        y_rank_score = 15
    elif y_rank == 3:
        y_rank_score = 10
    else:
        y_rank_score = 0
else:
    y_rank_score = 0
```

#### Change 3: Updated total score calculation (line ~1745)

```python
total_score = position_score + text_score + y_score + height_score + y_rank_score
```

#### Change 4: Added y_rank to breakdown dictionary (line ~1748)

```python
breakdown = {
    'position': position_score,
    'text': text_score,
    'y_position': y_score,
    'height': height_score,
    'y_rank': y_rank_score,  # ← NEW
    'total': total_score
}
```

#### Change 5: Calculate Y rankings before scoring (line ~1850)

```python
# Sort boxes by Y position to calculate rankings
sorted_top_boxes = sorted(top_boxes, key=lambda b: b.y_min)

# Create a mapping from box to its Y rank
box_to_rank = {id(box): rank + 1 for rank, box in enumerate(sorted_top_boxes)}
```

#### Change 6: Pass Y rank to scoring function (line ~1890)

```python
# Get Y rank for this box
y_rank = box_to_rank.get(id(box), None)

# Calculate score with Y rank
nickname_score, score_breakdown = self._calculate_nickname_score(
    box, cleaned_text, screen_width, screen_height, y_rank=y_rank
)
```

#### Change 7: Store Y rank in candidate data (line ~1900)

```python
candidates.append({
    'text': cleaned_text,
    'score': nickname_score,
    'score_breakdown': score_breakdown,
    'ocr_score': ocr_score,
    'box': box.box.tolist(),
    'center_x': box.center_x,
    'y_min': box.y_min,
    'y_rank': y_rank  # ← NEW
})
```

#### Change 8: Updated logging output (line ~1910)

```python
print(f"  '{cleaned_text}' -> score: {nickname_score:.1f} "
      f"(OCR: {ocr_score:.3f}, pos: {box.center_x:.0f}, y: {box.y_min:.0f}, rank: {y_rank})",  # ← ADDED rank
      file=log_file)
print(f"    Breakdown: pos={score_breakdown['position']:.1f}, "
      f"text={score_breakdown['text']:.1f}, "
      f"y={score_breakdown['y_position']:.1f}, "
      f"height={score_breakdown['height']:.1f}, "
      f"y_rank={score_breakdown['y_rank']:.1f}",  # ← ADDED y_rank
      file=log_file)
```

## Testing

### Test Results

Ran test with `test_images/test_whatsapp.png`:

```
Top Nickname: 'Gg Gg (你)'
Total Score: 78.8
Y Rank: 1

Score Breakdown:
  position    :  11.7
  text        :  30.0
  y_position  :  15.0
  height      :   2.1
  y_rank      :  20.0  ← Got 20 points for being rank 1
  total       :  78.8

Top 5 Candidates:
1. 'Gg Gg (你)' - Y Rank: 1 - y_rank=20.0
2. '给自己发消息' - Y Rank: 3 - y_rank=10.0
3. '00。' - Y Rank: 4 - y_rank=0.0
4. '今天' - Y Rank: 5 - y_rank=0.0
```

✓ Rank 1 correctly receives 20 points
✓ Rank 3 correctly receives 10 points
✓ Rank 4+ correctly receive 0 points

### Verification Script

Created `test_y_rank_simple.py` to verify implementation:
- ✓ All 10 checks passed
- ✓ Y-rank scoring properly integrated into codebase

## Impact

This change improves nickname detection accuracy by giving preference to text boxes that appear higher on the screen, which is where nicknames typically appear in chat applications.

The Y-rank score provides an additional 20 points (20% of total score) to help distinguish nicknames from other text elements.

## Files Modified

- `src/screenshotanalysis/processors.py` - Added Y-rank scoring logic

## Files Created (for testing)

- `test_y_rank_simple.py` - Verification script
- `test_y_rank_demo.py` - Demo script showing Y-rank scoring in action
- `Y_RANK_SCORING_IMPLEMENTATION.md` - This documentation

## Date

January 23, 2026
