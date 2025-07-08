-- 11 list all shows by total rating
SELECT
  ts.title,
  COALESCE(SUM(r.rating), 0) AS rating
FROM
  tv_shows ts
LEFT JOIN
  tv_show_ratings r ON ts.id = r.tv_show_id
GROUP BY
  ts.title
ORDER BY
  rating DESC;
