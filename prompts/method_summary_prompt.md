From Title and Text (Abstract+Segment), internally do 2 steps:
(1) Extract and compress ONLY explicitly stated methodological evidence (design/procedure, data/samples, tools/software, preprocessing, models/algorithms, parameters/variables, evaluation metrics) into ≤400 words.
(2) Using ONLY that evidence, write a method-focused abstract of about 220 words (target range 180–250 words), prioritizing methodological details (data source, sample scope, coding/classification procedure, variables, and analysis steps).

Rules:
- no judgment/classification/inference
- do NOT mention missing/unknown info
- keep variable names/parameters EXACTLY as written
- do NOT invent symbols/notation
- prefer concrete methodological details over background/context
- If the text contains little explicit methodological evidence, write a concise neutral summary of the segment content instead of returning None.

Output EXACTLY 2 lines:
title: <exact Title>
abs: <method-focused abstract OR concise neutral summary>

Title: {title}
Text: {abs_and_seg}