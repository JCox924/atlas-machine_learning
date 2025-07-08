-- 21 create SafeDiv function that returns a/b or 0 if b = 0
DELIMITER $$

DROP FUNCTION IF EXISTS SafeDiv$$

CREATE FUNCTION SafeDiv(
  a INT,
  b INT
)
RETURNS DOUBLE
DETERMINISTIC
BEGIN
  -- if divisor is zero, return 0; otherwise return the division
  IF b = 0 THEN
    RETURN 0;
  END IF;
  RETURN a / b;
END$$

DELIMITER ;
