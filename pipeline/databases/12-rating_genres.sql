-- 12 list all genres by total rating
SELECT
  g.name AS name,
  SUM(r.rating) AS rating
FROM
  tv_genres g
JOIN
  tv_show_genres sg ON g.id = sg.genre_id
JOIN
  tv_show_ratings r ON sg.tv_show_id = r.tv_show_id
GROUP BY
  g.name
ORDER BY
  rating DESC;
