# DynamicRules Evolution (v11, mini)

Компактный фрагмент для слайда (только ключевые изменения).

```diff
--- a/results_v11/al_iter_0/best_prompt.txt
+++ b/results_v11/al_iter_4/best_prompt.txt
@@ -11,3 +11,3 @@
-    STAR LEVEL CRITERIA:
+STAR BOUNDARIES:
@@ -23,5 +27,5 @@
-    3 Stars (Neutral / Mixed):
-    - Balanced pros and cons
-    - "It's okay," "average," "not great but not terrible"
-    - Works but with noticeable limitations
-    - If sentiment is clearly mixed and neither side dominates, choose 3
+3 STARS (Neutral/Mixed):
+- Balanced pros and cons, neither strongly positive nor negative
+- "Okay," "average," "works but with limitations," "better than nothing"
+- Mixed tone with notable complaints -> 3 even if some positive language present
+- Positive but with notable functional limitations that impact usability -> 3
@@ -41,5 +56,5 @@
-    EDGE CASE HANDLING:
-    - Mixed sentiment: Weigh overall conclusion more heavily than isolated complaints.
-    - Sarcasm: Detect implicit negativity even if positive words are used ironically.
-    - If safety risk is mentioned → lean toward 1.
-    - "Better than nothing" → usually 3.
+KEY DECISIONS:
+- If the review explicitly states a star rating, use that rating.
+- 4 vs 5: Focus on enthusiasm and whether criticisms affect satisfaction.
+- 3 vs 4: Notable functional limitations impacting usability -> 3; minor nitpicks -> 4.
+- Weigh overall conclusion more than isolated phrases.
```
