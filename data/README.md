# Synthetic smoke dataset

40 manually authored English tickets: 20 development and 20 test. Each split has
four tickets per department, including `other`. Labels reflect the primary issue;
equally important unrelated issues and insufficient context are `other` and need review.
Related topic groups do not cross splits. Some test rows include negation, mixed
issues and an instruction-injection attempt.

These examples check the pipeline, not production quality. They are small,
balanced and synthetic; they cannot establish calibration or represent real ticket
frequencies. Replace with a larger, independently labeled dataset before publishing
claims. Fix the annotation policy and split by conversation or customer group
before running any models. Do not tune prompts or thresholds on test rows.

Each JSONL record needs `id`, `group`, `split` (`dev` or `test`), `label` and `text`.
The dataset is included under this project's MIT license.

