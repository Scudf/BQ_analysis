SELECT
  t.task_id,
  s.overall_score,
  t.ama_query AS question,
  t.final_knowledge_card AS agent_response
FROM `ss-ees-bigquery-uat-297d.marketing_agent_assist_analytics_uat.aiaa_evaluation_data_v1` t
JOIN `ss-ees-bigquery-uat-297d.marketing_agent_assist_analytics_uat.aiaa_evaluation_scores_v1` s
  ON t.task_id = s.task_id
WHERE s.overall_score < {score_threshold}
  AND t.ama_query IS NOT NULL
LIMIT 5000;
