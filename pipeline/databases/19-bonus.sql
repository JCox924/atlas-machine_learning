-- 19-bonus.sql: create AddBonus stored procedure
DELIMITER $$

DROP PROCEDURE IF EXISTS AddBonus$$

CREATE PROCEDURE AddBonus(
  IN p_user_id INT,
  IN p_project_name VARCHAR(255),
  IN p_score INT
)
BEGIN
  DECLARE proj_id INT;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET proj_id = NULL;

  -- Try to find existing project
  SELECT id
    INTO proj_id
    FROM projects
   WHERE name = p_project_name
   LIMIT 1;

  -- If not found, create it
  IF proj_id IS NULL THEN
    INSERT INTO projects (name)
    VALUES (p_project_name);
    SET proj_id = LAST_INSERT_ID();
  END IF;

  -- Insert the bonus correction
  INSERT INTO corrections (user_id, project_id, score)
  VALUES (p_user_id, proj_id, p_score);
END$$

DELIMITER ;
