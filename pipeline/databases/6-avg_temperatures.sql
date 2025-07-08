-- 6 display the average temperature (Fahrenheit) by city, ordered by descending temperature
SELECT city,
       ROUND(AVG(temperature), 4) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;
