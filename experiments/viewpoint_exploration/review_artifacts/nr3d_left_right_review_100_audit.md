# Nr3D left/right first-batch audit

Input: 100 deterministic candidate queries from seed `20260920`.

Approved: 16. Inclusion required one explicit anchor that maps to one object-table instance, and a left/right relation directly linking the target and that anchor.

Excluded: 84. Exclusion reasons include no anchor, plural/group anchors, absolute image/room direction, ordinal or distance constraints, compound spatial descriptions, or an anchor absent from the object table.

The output manifest does not include `allowed_headings_deg`: the first experiment enumerates the full horizontal space. `target_id` is retained only for evaluation after geometric diagnosis.
