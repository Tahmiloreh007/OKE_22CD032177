CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    emotion TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
