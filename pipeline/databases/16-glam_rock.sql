-- 16 list all Glam rock bands ranked by their longevity (lifespan until 2020 in years)
SELECT
  band_name,
  CASE
    WHEN split IS NULL THEN 2020 - formed
    ELSE split - formed
  END AS lifespan
FROM
  metal_bands
WHERE
  main_style = 'Glam rock'
ORDER BY
  lifespan DESC;
